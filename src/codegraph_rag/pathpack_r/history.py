"""Git-history priors and commit memory for PathPack-E / PathPack-H.

Only commits reachable from ``base_commit`` are read: the snapshot HEAD is the task's ``base_commit``,
so ``git log`` cannot see the fix commit. That is the leakage guard. Nothing here reads a gold patch.

Commit memory: past commit messages are short descriptions of past changes. A new issue is matched
against them with BM25, and the files those commits touched receive a vote.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from ..retrieval import terms

MAX_COMMITS = 3000
MAX_FILES_PER_COMMIT = 25
MEMORY_TOP = 20
_FIX = re.compile(r'\b(fix|fixed|fixes|bug|regression|crash|error|incorrect|wrong)\b', re.I)
_VERSION = 2


class History:
    """Per-file churn, bug-fix counts, co-change counts and the commit list, newest first."""

    def __init__(self, commits=None, source='empty'):
        # commits: list of (hash, subject, [python files]), newest first
        self.commits = [(h, s, list(f)) for h, s, f in (commits or [])]
        self.source = source
        self.churn, self.fixes = Counter(), Counter()
        self.cochange = defaultdict(Counter)
        for _hash, subject, files in self.commits:
            if not files or len(files) > MAX_FILES_PER_COMMIT:
                continue
            is_fix = bool(_FIX.search(subject))
            for path in files:
                self.churn[path] += 1
                if is_fix:
                    self.fixes[path] += 1
            for a in files:
                for b in files:
                    if a != b:
                        self.cochange[a][b] += 1
        self._memory = None

    def file_churn(self, path):
        return self.churn.get(path, 0)

    def file_fixes(self, path):
        return self.fixes.get(path, 0)

    def cochange_with(self, path, anchors):
        """Mean share of each anchor's commits that also touched ``path``. Range [0, 1]."""
        total, used = 0.0, 0
        for anchor in anchors:
            if anchor == path:
                continue
            base = self.churn.get(anchor, 0)
            if base <= 0:
                continue
            total += self.cochange.get(anchor, Counter()).get(path, 0) / base
            used += 1
        return total / used if used else 0.0

    def after(self, commit_hash):
        """History strictly older than ``commit_hash``. Used by history replay so a commit never sees itself."""
        for i, (h, _s, _f) in enumerate(self.commits):
            if h == commit_hash:
                return History(self.commits[i + 1:], 'suffix')
        return History([], 'suffix_missing')

    # ---- commit memory -------------------------------------------------
    def _build_memory(self):
        docs = [Counter(terms(subject)) for _h, subject, _f in self.commits]
        df = Counter(word for doc in docs for word in doc)
        lengths = [sum(doc.values()) for doc in docs]
        avg = (sum(lengths) / len(lengths)) if lengths else 1.0
        self._memory = (docs, df, lengths, avg)

    def memory_scores(self, query, top=MEMORY_TOP):
        """Return ``{path: score in [0, 1]}`` from the ``top`` most similar past commit messages."""
        if not self.commits:
            return {}
        if self._memory is None:
            self._build_memory()
        docs, df, lengths, avg = self._memory
        words = sorted(set(terms(query)))
        n = len(docs)
        k1, b = 1.5, 0.75
        scored = []
        for i, doc in enumerate(docs):
            files = self.commits[i][2]
            if not files or len(files) > MAX_FILES_PER_COMMIT:
                continue
            score = 0.0
            for word in words:
                tf = doc.get(word)
                if not tf:
                    continue
                idf = math.log(1 + (n - df[word] + 0.5) / (df[word] + 0.5))
                score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lengths[i] / max(avg, 1)))
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda item: (-item[0], item[1]))
        votes = Counter()
        for score, i in scored[:top]:
            for path in self.commits[i][2]:
                votes[path] += score / len(self.commits[i][2])
        best = max(votes.values(), default=0.0)
        return {path: value / best for path, value in votes.items()} if best > 0 else {}

    def to_json(self):
        return {'version': _VERSION, 'commits': self.commits}


def parse_log(text):
    """Parse ``git log --name-only --format=@@<TAB><hash><TAB><subject>`` output."""
    commits = []
    current = None
    for line in text.splitlines():
        if line.startswith('@@\t'):
            if current is not None:
                commits.append(current)
            parts = line.split('\t', 2)
            current = (parts[1], parts[2] if len(parts) > 2 else '', [])
        elif line.strip() and current is not None and line.strip().endswith('.py'):
            current[2].append(line.strip())
    if current is not None:
        commits.append(current)
    commits = [(h, s, sorted(set(f))) for h, s, f in commits]
    return History(commits, 'git_log')


def load_history(repository, *, cache_dir=None, cache_key=None):
    """Return a History for the snapshot at HEAD. Cached on disk by ``cache_key``."""
    cache = None
    if cache_dir and cache_key:
        cache = Path(cache_dir) / f'history{_VERSION}_{cache_key}.json'
        if cache.is_file():
            return History(json.loads(cache.read_text())['commits'], 'cache')
    try:
        out = subprocess.run(
            ['git', '-C', str(repository), 'log', '-n', str(MAX_COMMITS), '--no-merges', '--name-only',
             '--format=@@%x09%H%x09%s'],
            check=True, capture_output=True, text=True, timeout=180,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return History(source='unavailable')
    history = parse_log(out)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(history.to_json()))
    return history
