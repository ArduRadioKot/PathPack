"""Line-level gold metrics that do not change Changed Line Recall.

Changed Line Recall stays in ``evaluation_v4.score_prediction``. It is the
share of non-empty base-file lines deleted or replaced by the patch (``-``
lines). Pure insertions have no such line. They stay out of that denominator,
and a task whose patch only inserts lines has Changed Line Recall = null.
Added text is not searched for in ``base_commit``: it is not in the repository
the retriever can see.

Insertion-anchor recall is a separate metric. For a hunk that adds lines and
deletes none, the insertion site is the base line before the added block and
the base line after it. If that bracket line is empty, one adjacent non-empty
line is used and the search stops. A zero-length old hunk (``@@ -N,0``) anchors
on line N, the line after which Git inserts. A site that still does not resolve
inside the base file is excluded with ``insertion_anchor_unmapped`` and is not
folded into Changed Line Recall. New files are ``new_file_no_base_source``.

Base-touch recall is the explicit union of evaluable changed lines and resolved
anchors. It is never reported under the name Changed Line Recall.

Relevant Line Density, for a task that has at least one evaluable changed line
and a non-empty context, is

    |emitted non-empty lines that are evaluable changed lines| / |emitted non-empty lines|

Lines outside that changed-line set are treated as not relevant because the
gold patch is the complete edit. Tasks without evaluable changed lines are
excluded (null), not scored as zero. An empty context is null, reason
``empty_context``. Insertion anchors do not count as relevant for this density:
they are context around an addition, not the changed line itself. Density
penalizes extra explanatory lines, so it is reported beside recall and is not
a target to maximize on its own.
"""
from __future__ import annotations

from collections import defaultdict
import re

from .evaluation_v4 import covered_lines, label_patch

_HUNK=re.compile(r'^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@')


def evaluable_changed(gold):
    source=gold['source_lines']
    return {(path,line) for path,lines in gold['changed_lines'].items() for line in lines
            if line<=len(source.get(path,[])) and source[path][line-1].strip()}


def _parse_files(patch):
    files=[];current=None;hunk=None
    def close():
        nonlocal hunk
        if hunk is not None and current is not None:
            current['hunks'].append(hunk)
        hunk=None
    for text in patch.splitlines():
        if text.startswith('diff --git '):
            close();current=None
        elif text.startswith('--- '):
            close()
            path=text[6:] if text.startswith('--- a/') else None
            current={'path':path,'hunks':[]}
            files.append(current)
        elif text.startswith('@@ ') and current is not None:
            close()
            match=_HUNK.match(text)
            old=int(match.group(1)) if match else 0
            old_len=int(match.group(2)) if match and match.group(2) is not None else 1
            hunk={'old':old,'old_len':old_len,'lines':[]}
        elif hunk is not None and (text[:1] in '+- '):
            hunk['lines'].append(text)
    close()
    return files


def _sites(hunk):
    """Bracketing base lines for each addition group. Removals yield no site."""
    if any(line.startswith('-') for line in hunk['lines']):
        return []
    if not any(line.startswith('+') for line in hunk['lines']):
        return []
    old=hunk['old'];pred=None;sites=[];in_plus=False;group_pred=None
    for line in hunk['lines']:
        if line.startswith('+'):
            if not in_plus:
                group_pred=pred;in_plus=True
        else:
            if in_plus:
                sites.append((group_pred,old));in_plus=False
            pred=old;old+=1
    if in_plus:
        sites.append((group_pred,None))
    if hunk['old_len']==0 and sites==[(None,None)]:
        sites=[(hunk['old'] if hunk['old']>=1 else None,None)]
    return sites


def _resolve_side(lines,number,direction):
    if number is None or not 1<=number<=len(lines):
        return None
    if lines[number-1].strip():
        return number
    shifted=number+direction
    if 1<=shifted<=len(lines) and lines[shifted-1].strip():
        return shifted
    return None


def insertion_anchors(patch,source_lines):
    """Return ``{path: set(line numbers)}`` and exclusion records."""
    anchors=defaultdict(set);exclusions=[];hunks=0
    for current in _parse_files(patch):
        path=current['path']
        if path is None:
            if any(h['lines'] for h in current['hunks']):
                exclusions.append({'path':None,'reason':'new_file_no_base_source'})
            continue
        lines=source_lines.get(path)
        if lines is None:
            exclusions.append({'path':path,'reason':'missing_base_file'});continue
        for hunk in current['hunks']:
            sites=_sites(hunk)
            if not sites:
                continue
            hunks+=1
            resolved=[]
            for pred,succ in sites:
                chosen=[]
                before=_resolve_side(lines,pred,-1)
                after=_resolve_side(lines,succ,+1)
                if before is not None:
                    chosen.append(before)
                if after is not None and after not in chosen:
                    chosen.append(after)
                if chosen:
                    resolved.extend(chosen)
                else:
                    exclusions.append({'path':path,'reason':'insertion_anchor_unmapped','old':hunk['old']})
            if resolved:
                anchors[path].update(resolved)
    return anchors,exclusions,hunks


def build_line_label(patch,repository):
    """Official v0.4 label plus the separate insertion-anchor layer."""
    gold=label_patch(patch,repository)
    anchors,exclusions,hunks=insertion_anchors(patch,gold['source_lines'])
    changed=evaluable_changed(gold)
    bodies={(fn['path'],line) for fn in gold['functions'].values() for line in fn['body_lines']}
    outside=sorted(changed-bodies)
    out_of_range=[]
    for path,lines in gold['changed_lines'].items():
        length=len(gold['source_lines'].get(path,[]))
        out_of_range.extend({'path':path,'line':line} for line in lines if line<1 or line>length)
    nested=[];methods=[]
    for fn in gold['functions'].values():
        if fn['qualname'].count('.')>=2:
            nested.append(fn['id'])
        elif '.' in fn['qualname']:
            methods.append(fn['id'])
    notes={'changed_line_definition':'nonempty_removed_or_replaced_base_lines',
           'insertion_anchor_definition':'bracket_base_lines_of_addition_only_hunks_blank_skip_1',
           'metrics_are_mixed':False,
           'added_lines_are_absent_from_base':True,
           'insertion_hunks':hunks,
           'changed_lines_outside_function_bodies':[{'path':path,'line':line} for path,line in outside],
           'removed_lines_out_of_range':out_of_range,
           'class_methods':methods,'nested_functions':nested,
           'module_level_exclusions':sum(item.get('reason')=='module_level_change' for item in gold['exclusions']),
           'new_definition_exclusions':sum(item.get('reason')=='new_definition_no_base_function' for item in gold['exclusions'])}
    return {**gold,'insertion_anchors':{path:sorted(lines) for path,lines in anchors.items()},
            'anchor_exclusions':exclusions,'line_audit':notes}


def extra_metrics(selected,gold):
    covered=covered_lines(selected,gold['source_lines'])
    changed=evaluable_changed(gold)
    anchors={(path,line) for path,lines in gold.get('insertion_anchors',{}).items() for line in lines}
    exclusions=gold.get('anchor_exclusions') or []
    if not changed:
        density=None;density_reason='no_evaluable_changed_lines'
    elif not covered:
        density=None;density_reason='empty_context'
    else:
        density=len(changed & covered)/len(covered);density_reason=None
    if anchors:
        anchor_recall=len(anchors & covered)/len(anchors)
        anchor_reason='partial_anchor_mapping' if exclusions else None
    elif exclusions:
        anchor_recall=None;anchor_reason=exclusions[0]['reason']
    else:
        anchor_recall=None;anchor_reason='no_insertion_hunk'
    union=changed|anchors
    if union:
        base_touch=len(union & covered)/len(union);base_reason=None
    elif exclusions and not changed:
        base_touch=None;base_reason=exclusions[0]['reason']
    else:
        base_touch=None;base_reason='no_base_touch_lines'
    return {'insertion_anchor_recall':anchor_recall,'insertion_anchor_reason':anchor_reason,
            'base_touch_recall':base_touch,'base_touch_reason':base_reason,
            'relevant_line_density':density,'relevant_line_density_reason':density_reason,
            'relevant_line_density_definition':'changed_line_hits_over_emitted_nonempty_lines',
            'evaluable_changed_lines':len(changed),'evaluable_anchor_lines':len(anchors),
            'emitted_nonempty_lines':len(covered)}
