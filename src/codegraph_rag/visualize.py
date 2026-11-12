"""Draw annotated, reproducible graphs from CodeGraph-RAG JSON/objects.

Visualization is illustrative. Graphs are static approximations of Python code.
"""
from __future__ import annotations

import json
from pathlib import Path


def select_graph(data: dict, focus: str | None = None, hops: int = 1,
                 edge_kinds: set[str] | None = None, max_nodes: int = 40) -> dict:
    """Keep a stable subset for legible visualization, never change source graph."""
    import networkx as nx
    nodes = {n["id"]: n for n in data["nodes"]}
    edges = [e for e in data["edges"] if e["source"] in nodes and e["target"] in nodes
             and (edge_kinds is None or e["kind"] in edge_kinds)]
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from((e["source"], e["target"]) for e in edges)
    if focus:
        hits = [n for n in nodes if focus.lower() in n.lower()]
        if not hits:
            raise ValueError(f"Cannot find a node matching {focus!r}")
        # Prefer an exact ID, otherwise the shortest matching ID.
        start = sorted(hits, key=lambda n: (n != focus, len(n), n))[0]
        distance = nx.single_source_shortest_path_length(graph, start, cutoff=hops)
        keep = set(sorted(distance, key=lambda x: (distance[x], x))[:max_nodes])
    else:
        keep = set(nodes) if len(nodes) <= max_nodes else set(
            sorted(nodes, key=lambda x: (-graph.degree[x], x))[:max_nodes])
    return {"nodes": [nodes[n] for n in sorted(keep)],
            "edges": [e for e in edges if e["source"] in keep and e["target"] in keep]}


def draw_graph(data: dict, output: str | Path | None = None,
               *, title: str = "Python repository dependency graph",
               focus: str | None = None, hops: int = 1,
               edge_kinds: set[str] | None = None, max_nodes: int = 40,
               figsize: tuple[float, float] = (12, 7)):
    """Draw graph with labelled nodes and typed directed edges, return Matplotlib Figure."""
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch, Patch

    sliced = select_graph(data, focus=focus, hops=hops,
                          edge_kinds=edge_kinds, max_nodes=max_nodes)
    nodes = {n["id"]: n for n in sliced["nodes"]}
    edges = sliced["edges"]
    colors = {"module": "#27374D", "function": "#167D8D", "method": "#287FB8", "class": "#9563A8"}
    rel_colors = {"calls": "#17788B", "defines": "#8696A9", "imports": "#8867A9", "inherits": "#D28D38"}
    rel_styles = {"calls": "solid", "defines": "solid", "imports": "dashed", "inherits": "dotted"}
    H = nx.Graph()
    H.add_nodes_from(nodes)
    H.add_edges_from((e["source"], e["target"]) for e in edges)
    positions = nx.spring_layout(H, seed=12, k=1.8/max(len(nodes)**0.5,1), iterations=140)
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")
    edge_pairs = [(e["source"], e["target"]) for e in edges]
    for e in sorted(edges, key=lambda x: (x["kind"], x["source"], x["target"])):
        source, target = positions[e["source"]], positions[e["target"]]
        if e["source"] == e["target"]:
            continue
        # Slight curvature helps with overlapping directed edges.
        reverse = (e["target"],e["source"]) in edge_pairs
        rad = 0.16 if reverse else 0.05
        patch = FancyArrowPatch(source, target, arrowstyle="-|>", mutation_scale=12,
                                connectionstyle=f"arc3,rad={rad}",
                                shrinkA=32, shrinkB=32,
                                color=rel_colors.get(e["kind"], "#64748B"),
                                linestyle=rel_styles.get(e["kind"], "solid"),
                                linewidth=2.0 if e["kind"] == "calls" else 1.35,
                                alpha=0.82, zorder=1)
        ax.add_patch(patch)
    font_size=8 if len(nodes)>16 else 10
    for node_id,n in nodes.items():
        x,y=positions[node_id]
        if n["kind"] == "module":
            label=Path(n["path"]).name+"\n[module]"
        else:
            label=n["qualname"].split(".")[-1]
            if n["kind"] in {"function", "method"}: label += "()"
            label += "\n["+n["kind"]+"]"
        ax.text(x, y, label, fontsize=font_size, color="white", ha="center", va="center",
                linespacing=1.25, zorder=3,
                bbox=dict(boxstyle="round,pad=0.54,rounding_size=0.28",
                          facecolor=colors.get(n["kind"], "#445"),
                          edgecolor="#FFFFFF", linewidth=1.4))
    ax.set_xlim(min(x for x,y in positions.values())-0.43, max(x for x,y in positions.values())+0.43)
    ax.set_ylim(min(y for x,y in positions.values())-0.38, max(y for x,y in positions.values())+0.34)
    ax.axis("off")
    ax.set_title(title, fontsize=17, color="#182D45", loc="left", weight="bold", pad=19)
    leg1 = [Patch(facecolor=c, label=k) for k,c in colors.items() if any(n["kind"]==k for n in nodes.values())]
    leg2 = [Line2D([0],[0],color=c,linestyle=rel_styles[k],linewidth=2,label=k)
            for k,c in rel_colors.items() if any(e["kind"]==k for e in edges)]
    fig.legend(handles=leg1+leg2, loc="lower center", bbox_to_anchor=(.5,.02),
               ncol=min(8,len(leg1+leg2)), frameon=False, fontsize=9)
    fig.subplots_adjust(left=.02,right=.98,top=.9,bottom=.13)
    if output is not None:
        dest = Path(output)
        dest.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(dest, dpi=190, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig


def visualize_file(source: str | Path, output: str | Path, **kwargs):
    data=json.loads(Path(source).read_text(encoding="utf-8"))
    return draw_graph(data,output,**kwargs)
