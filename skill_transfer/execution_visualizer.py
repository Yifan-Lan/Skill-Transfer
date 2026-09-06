"""
Visualize an ExecutionStructure (or a pair) as a Graphviz graph.

Exported functions:
    visualize_structure(structure, output_path, ...)  → single agent graph
    visualize_comparison(strong, weak, output_path, ...) → side-by-side graph

Output formats: SVG (default), PNG, PDF — inferred from output_path suffix.
"""

from __future__ import annotations

import html
import textwrap
from pathlib import Path
from typing import Optional

import graphviz

from execution_structure import ExecutionStructure, GapReport, ProcedureNode, Tier1Type

# ──────────────────────────────────────────────────────────────────────────────
# Colour palette  (background, header)
# ──────────────────────────────────────────────────────────────────────────────

TIER1_PALETTE: dict[str, tuple[str, str]] = {
    "setup":    ("#D6EAF8", "#2E86C1"),   # blue family
    "execute":  ("#D5F5E3", "#1E8449"),   # green family
    "validate": ("#FDEBD0", "#CA6F1E"),   # orange family
    "recover":  ("#FADBD8", "#C0392B"),   # red family
    "output":   ("#E8DAEF", "#7D3C98"),   # purple family
    "other":    ("#EAECEE", "#5D6D7E"),   # grey family
}

OUTCOME_COLORS = {
    "success": ("#1E8449", "✓"),
    "error":   ("#C0392B", "✗"),
    "partial": ("#D4AC0D", "~"),
    "skipped": ("#7F8C8D", "–"),
}


# ──────────────────────────────────────────────────────────────────────────────
# HTML label helpers
# ──────────────────────────────────────────────────────────────────────────────

def _esc(text: str) -> str:
    """Escape text for use inside Graphviz HTML labels."""
    return html.escape(str(text), quote=False)


def _wrap(text: str, width: int = 38) -> str:
    """Wrap long text into multiple lines for the label."""
    lines = textwrap.wrap(text, width=width)
    return "<BR/>".join(_esc(l) for l in lines) if lines else ""


def _node_label(node: ProcedureNode, show_dataflow: bool = True) -> str:
    """Build an HTML table label for a single ProcedureNode."""
    bg, header_color = TIER1_PALETTE.get(node.category_tier1, TIER1_PALETTE["other"])

    # Outcome badge
    oc_color, oc_icon = OUTCOME_COLORS.get(node.outcome, ("#7F8C8D", "?"))

    # Tier2 label (main title)
    title = _esc(node.category_tier2.replace("_", " "))

    # Tier1 badge + outcome
    tier1_badge = _esc(node.category_tier1.upper())
    step_info = f"step {node.step_index}"

    # completes_when (wrapped, max 2 lines ~ 76 chars)
    cw_text = node.completes_when
    cw_lines = textwrap.wrap(cw_text, width=40)[:2]
    if len(textwrap.wrap(cw_text, width=40)) > 2:
        cw_lines[-1] = cw_lines[-1].rstrip(".") + "…"
    cw_html = "<BR/>".join(_esc(l) for l in cw_lines)

    # produces / consumes
    produces_str = ", ".join(node.produces) if node.produces else "—"
    consumes_str = ", ".join(node.consumes) if node.consumes else "—"

    # Build HTML label
    label_parts = [
        # Header bar (tier1 color)
        f'<TR>'
        f'<TD BGCOLOR="{header_color}" ALIGN="CENTER" COLSPAN="2">'
        f'<FONT COLOR="white" POINT-SIZE="8"><B>{tier1_badge}</B>  '
        f'<FONT COLOR="#DDDDDD">{step_info}</FONT></FONT>'
        f'</TD>'
        f'</TR>',

        # Main title (tier2 label)
        f'<TR>'
        f'<TD BGCOLOR="{bg}" ALIGN="CENTER" COLSPAN="2" CELLPADDING="5">'
        f'<FONT POINT-SIZE="11"><B>{title}</B></FONT>'
        f'</TD>'
        f'</TR>',

        # completes_when
        f'<TR>'
        f'<TD BGCOLOR="white" ALIGN="LEFT" COLSPAN="2" CELLPADDING="4">'
        f'<FONT POINT-SIZE="8" COLOR="#444444">'
        f'<I>done when: {cw_html}</I>'
        f'</FONT>'
        f'</TD>'
        f'</TR>',
    ]

    if show_dataflow:
        label_parts += [
            # produces
            f'<TR>'
            f'<TD BGCOLOR="#F9F9F9" ALIGN="LEFT" CELLPADDING="3">'
            f'<FONT POINT-SIZE="8" COLOR="#1E8449">'
            f'<B>▶</B> {_esc(produces_str)}'
            f'</FONT>'
            f'</TD>'
            # outcome badge
            f'<TD BGCOLOR="#F9F9F9" ALIGN="RIGHT" CELLPADDING="3">'
            f'<FONT POINT-SIZE="9" COLOR="{oc_color}"><B>{oc_icon}</B></FONT>'
            f'</TD>'
            f'</TR>',
        ]

        if node.consumes:
            label_parts.append(
                f'<TR>'
                f'<TD BGCOLOR="#F9F9F9" ALIGN="LEFT" COLSPAN="2" CELLPADDING="3">'
                f'<FONT POINT-SIZE="8" COLOR="#7D3C98">'
                f'<B>◀</B> {_esc(consumes_str)}'
                f'</FONT>'
                f'</TD>'
                f'</TR>'
            )
    else:
        # Just show outcome badge
        label_parts.append(
            f'<TR>'
            f'<TD BGCOLOR="#F9F9F9" ALIGN="RIGHT" COLSPAN="2" CELLPADDING="3">'
            f'<FONT POINT-SIZE="9" COLOR="{oc_color}"><B>{oc_icon}</B></FONT>'
            f'</TD>'
            f'</TR>'
        )

    rows = "\n".join(label_parts)
    return f'<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="0">\n{rows}\n</TABLE>>'


def _weak_node_label(node: ProcedureNode, gap: "GapEntry", show_dataflow: bool) -> str:
    """Like _node_label but adds an orange warning row showing the gap description."""
    base = _node_label(node, show_dataflow=show_dataflow)
    # base ends with </TABLE>> — insert warning row before the closing tag
    desc_short = gap.description[:70] + "…" if len(gap.description) > 70 else gap.description
    warning_row = (
        f'<TR><TD BGCOLOR="#FEF9E7" ALIGN="LEFT" COLSPAN="2" CELLPADDING="3">'
        f'<FONT POINT-SIZE="7" COLOR="#CA6F1E">⚠ weak: {_esc(desc_short)}</FONT>'
        f'</TD></TR>'
    )
    # Insert before the final </TABLE>>
    return base[:-len("</TABLE>>")] + warning_row + "</TABLE>>"


def _add_structure_nodes(
    g: graphviz.Digraph,
    structure: ExecutionStructure,
    prefix: str = "",
    highlighted_tier1: Optional[set[str]] = None,
    dim: bool = False,
    show_dataflow: bool = True,
    weak_node_gaps: Optional[dict[str, "GapEntry"]] = None,
) -> None:
    """Add all nodes and edges from an ExecutionStructure to graph g.

    prefix            : prepended to node ids to avoid collisions in combined graphs.
    highlighted_tier1 : nodes with these tier1 values get a bold border (for strong cluster).
    dim               : render nodes faded (unused currently, reserved for future).
    show_dataflow     : whether to include produces/consumes rows in the label.
    weak_node_gaps    : mapping of {node.id → GapEntry} for weak_node gaps; those nodes
                        get an orange warning annotation appended to their label.
    """
    nodes = structure.nodes
    weak_node_gaps = weak_node_gaps or {}

    for node in nodes:
        node_id = f"{prefix}{node.id}"

        # Weak-node gap annotation takes priority
        if node.id in weak_node_gaps:
            gap = weak_node_gaps[node.id]
            label = _weak_node_label(node, gap, show_dataflow=show_dataflow)
            g.node(
                node_id,
                label=label,
                shape="plaintext",
                tooltip=f"⚠ weak node: {gap.description}",
                color="#CA6F1E",
                penwidth="2",
            )
            continue

        label = _node_label(node, show_dataflow=show_dataflow)
        attrs: dict[str, str] = {
            "shape": "plaintext",
            "tooltip": f"{node.category_tier1}: {node.completes_when}",
        }

        if highlighted_tier1 and node.category_tier1 in highlighted_tier1:
            # Bold border on strong-cluster nodes whose tier1 is absent in weak
            attrs["shape"] = "rectangle"
            attrs["style"] = "bold,filled"
            attrs["fillcolor"] = "white"
            attrs["color"] = TIER1_PALETTE[node.category_tier1][1]
            attrs["penwidth"] = "3"
            plain = f'{node.category_tier2}\n[{node.category_tier1}]'
            g.node(node_id, label=plain, **attrs)
            continue

        g.node(node_id, label=label, **attrs)

    # Sequential flow edges
    for i in range(len(nodes) - 1):
        src = f"{prefix}{nodes[i].id}"
        dst = f"{prefix}{nodes[i + 1].id}"
        # Skip if next node is a branch_of (will draw separately)
        if nodes[i + 1].branch_of == nodes[i].id:
            continue
        g.edge(src, dst, color="#555555", penwidth="1.5")

    # branch_of edges (dashed, coloured by recover/fallback)
    for node in nodes:
        if node.branch_of:
            src = f"{prefix}{node.branch_of}"
            dst = f"{prefix}{node.id}"
            g.edge(
                src, dst,
                style="dashed",
                color=TIER1_PALETTE["recover"][1],
                penwidth="1.5",
                xlabel=f'  {node.category_tier2.replace("_"," ")}  ',
                fontsize="8",
                fontcolor=TIER1_PALETTE["recover"][1],
                constraint="false",  # don't affect rank layout
            )
            # Also draw the sequential edge that skips the branch node
            parent_idx = next(n.step_index for n in nodes if n.id == node.branch_of)
            after_branch = [n for n in nodes if n.step_index == node.step_index + 1
                           and not n.branch_of]
            for nb in after_branch:
                g.edge(
                    f"{prefix}{node.id}",
                    f"{prefix}{nb.id}",
                    color="#555555", penwidth="1.5",
                )


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def visualize_structure(
    structure: ExecutionStructure,
    output_path: Path,
    show_dataflow: bool = True,
    title: Optional[str] = None,
) -> Path:
    """Render a single ExecutionStructure as a Graphviz graph.

    Args:
        structure:     The ExecutionStructure to visualise.
        output_path:   Destination path.  Format inferred from suffix
                       (.svg, .png, .pdf).  The suffix is stripped before
                       passing to graphviz (it adds its own).
        show_dataflow: Whether to include produces/consumes in node labels.
        title:         Optional graph title.

    Returns:
        Path to the rendered file (with format suffix appended by graphviz).
    """
    output_path = Path(output_path)
    fmt = output_path.suffix.lstrip(".") or "svg"
    base = output_path.with_suffix("")  # graphviz appends the suffix itself

    display_title = title or f"{structure.agent_id} — {structure.outcome}"

    g = graphviz.Digraph(
        name=structure.agent_id,
        graph_attr={
            "rankdir": "TB",
            "splines": "ortho",
            "nodesep": "0.4",
            "ranksep": "0.6",
            "bgcolor": "white",
            "label": _esc(display_title),
            "labelloc": "t",
            "fontsize": "14",
            "fontname": "Helvetica",
        },
        node_attr={"fontname": "Helvetica"},
        edge_attr={"fontname": "Helvetica"},
    )

    _add_structure_nodes(g, structure, show_dataflow=show_dataflow)

    # Legend
    _add_legend(g)

    rendered = g.render(str(base), format=fmt, cleanup=True)
    return Path(rendered)


def visualize_comparison(
    strong: ExecutionStructure,
    weak: ExecutionStructure,
    output_path: Path,
    gap_report: Optional[GapReport] = None,
    show_dataflow: bool = True,
    title: Optional[str] = None,
) -> Path:
    """Render strong and weak agent structures side-by-side in one graph.

    If gap_report is provided, gaps are annotated directly on the graph:
      - missing_node gaps: ghost nodes in weak cluster + labelled gap arrows
      - weak_node gaps: the weak agent's node gets an orange warning border

    Returns:
        Path to the rendered file.
    """
    output_path = Path(output_path)
    fmt = output_path.suffix.lstrip(".") or "svg"
    base = output_path.with_suffix("")

    display_title = title or (
        f"Structural Comparison: {strong.agent_id} (strong) vs {weak.agent_id} (weak)"
    )

    strong_only_tier1 = strong.tier1_categories() - weak.tier1_categories()

    # Build lookup maps from gap report
    missing_gaps: dict[str, "GapEntry"] = {}   # tier1 → GapEntry
    weak_node_gaps: dict[str, "GapEntry"] = {}  # weak_node_id → GapEntry
    if gap_report:
        for gap in gap_report.gaps:
            if gap.gap_type == "missing_node":
                missing_gaps[gap.category_tier1] = gap
            elif gap.gap_type == "weak_node" and gap.weak_node_ref:
                if isinstance(gap.weak_node_ref, dict):
                    for node_id in gap.weak_node_ref.values():
                        if node_id:
                            weak_node_gaps[node_id] = gap
                else:
                    weak_node_gaps[gap.weak_node_ref] = gap

    g = graphviz.Digraph(
        name="comparison",
        graph_attr={
            "rankdir": "TB",
            "splines": "ortho",
            "nodesep": "0.5",
            "ranksep": "0.7",
            "bgcolor": "#FAFAFA",
            "label": _esc(display_title),
            "labelloc": "t",
            "fontsize": "15",
            "fontname": "Helvetica",
            "compound": "true",
        },
        node_attr={"fontname": "Helvetica"},
        edge_attr={"fontname": "Helvetica"},
    )

    # ── Strong agent cluster ──────────────────────────────────────────────
    strong_label = (
        f'<<B>{_esc(strong.agent_id)}</B>  '
        f'<FONT COLOR="#1E8449">outcome: {strong.outcome}</FONT>  '
        f'<FONT COLOR="#555555">({len(strong.nodes)} steps, '
        f'{len(strong.produced_files)} files)</FONT>>'
    )
    with g.subgraph(name="cluster_strong") as cs:
        cs.attr(
            label=strong_label,
            style="rounded,filled",
            fillcolor="#F0FFF0",
            color="#1E8449",
            penwidth="2",
        )
        _add_structure_nodes(
            cs, strong, prefix="S_",
            highlighted_tier1=strong_only_tier1,
            show_dataflow=show_dataflow,
        )

    # ── Weak agent cluster ────────────────────────────────────────────────
    weak_color = "#C0392B" if weak.outcome in ("error", "partial") else "#CA6F1E"
    weak_label = (
        f'<<B>{_esc(weak.agent_id)}</B>  '
        f'<FONT COLOR="{weak_color}">outcome: {weak.outcome}</FONT>  '
        f'<FONT COLOR="#555555">({len(weak.nodes)} steps, '
        f'{len(weak.produced_files)} files)</FONT>>'
    )
    with g.subgraph(name="cluster_weak") as cw:
        cw.attr(
            label=weak_label,
            style="rounded,filled",
            fillcolor="#FFF8F0",
            color="#CA6F1E",
            penwidth="2",
        )
        _add_structure_nodes(
            cw, weak, prefix="W_",
            show_dataflow=show_dataflow,
            weak_node_gaps=weak_node_gaps,
        )
        # Ghost nodes for missing tier1 categories
        for t1 in sorted(strong_only_tier1):
            ghost_id = f"W_ghost_{t1}"
            bg, hc = TIER1_PALETTE.get(t1, TIER1_PALETTE["other"])
            # Include patch hint snippet if available
            gap = missing_gaps.get(t1)
            hint_line = ""
            if gap:
                hint_short = gap.skill_patch_hint[:55] + "…" if len(gap.skill_patch_hint) > 55 else gap.skill_patch_hint
                hint_line = (
                    f'<TR><TD BGCOLOR="white" ALIGN="LEFT" CELLPADDING="3">'
                    f'<FONT POINT-SIZE="7" COLOR="#555555"><I>{_esc(hint_short)}</I></FONT>'
                    f'</TD></TR>'
                )
            cw.node(
                ghost_id,
                label=(
                    f'<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">'
                    f'<TR><TD BGCOLOR="{hc}"><FONT COLOR="white" POINT-SIZE="8">'
                    f'<B>{_esc(t1.upper())}</B></FONT></TD></TR>'
                    f'<TR><TD BGCOLOR="{bg}"><FONT POINT-SIZE="9" COLOR="#888888">'
                    f'<B>MISSING</B></FONT></TD></TR>'
                    f'{hint_line}'
                    f'</TABLE>>'
                ),
                shape="plaintext",
                style="dashed",
                color=hc,
                penwidth="2",
                tooltip=gap.skill_patch_hint if gap else f"Missing: {t1}",
            )

    # ── Cross-cluster gap arrows ──────────────────────────────────────────
    for t1 in sorted(strong_only_tier1):
        strong_nodes = strong.nodes_by_tier1(t1)
        if not strong_nodes:
            continue
        gap = missing_gaps.get(t1)
        severity_color = "#C0392B" if gap and gap.severity == "high" else "#CA6F1E"
        gap_label = f"  {gap.gap_id} ({gap.severity})  " if gap else "  gap  "
        src = f"S_{strong_nodes[0].id}"
        dst = f"W_ghost_{t1}"
        g.edge(
            src, dst,
            style="dashed",
            color=severity_color,
            penwidth="1.5",
            arrowhead="open",
            xlabel=gap_label,
            fontsize="8",
            fontcolor=severity_color,
            ltail="cluster_strong",
            lhead="cluster_weak",
            constraint="false",
        )

    _add_legend(g, gap_report=gap_report)

    rendered = g.render(str(base), format=fmt, cleanup=True)
    return Path(rendered)


# ──────────────────────────────────────────────────────────────────────────────
# Legend
# ──────────────────────────────────────────────────────────────────────────────

def _add_legend(
    g: graphviz.Digraph,
    gap_report: Optional[GapReport] = None,
) -> None:
    """Add a compact tier1 colour legend, plus a gap summary if gap_report given."""
    # Tier-1 colour key
    color_rows = []
    for tier1, (bg, hc) in TIER1_PALETTE.items():
        color_rows.append(
            f'<TR>'
            f'<TD BGCOLOR="{hc}" WIDTH="10" HEIGHT="10"></TD>'
            f'<TD ALIGN="LEFT"><FONT POINT-SIZE="8"> {_esc(tier1)}</FONT></TD>'
            f'</TR>'
        )

    # Edge key
    edge_rows = [
        '<TR><TD COLSPAN="2"><FONT POINT-SIZE="8">─── sequential flow</FONT></TD></TR>',
        '<TR><TD COLSPAN="2"><FONT POINT-SIZE="8" COLOR="#C0392B">- - recover/fallback</FONT></TD></TR>',
    ]
    if gap_report:
        edge_rows.append(
            '<TR><TD COLSPAN="2"><FONT POINT-SIZE="8" COLOR="#C0392B">⚡ gap arrow</FONT></TD></TR>'
        )

    legend_label = (
        '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2" CELLPADDING="2">'
        '<TR><TD COLSPAN="2" ALIGN="CENTER">'
        '<FONT POINT-SIZE="9"><B>Tier-1</B></FONT></TD></TR>'
        + "".join(color_rows)
        + '<TR><TD COLSPAN="2"><FONT POINT-SIZE="1"> </FONT></TD></TR>'
        + "".join(edge_rows)
        + '</TABLE>>'
    )

    with g.subgraph(name="cluster_legend") as leg:
        leg.attr(label="", style="rounded", color="#CCCCCC", bgcolor="#FAFAFA")
        leg.node("_legend", label=legend_label, shape="plaintext", margin="0")

    # Gap summary panel (only when gap_report provided)
    if gap_report and gap_report.gaps:
        sev_color = {"high": "#C0392B", "medium": "#CA6F1E", "low": "#7F8C8D"}
        gap_rows = []
        for gap in gap_report.gaps:
            gtype_icon = "✗" if gap.gap_type == "missing_node" else "⚠"
            col = sev_color.get(gap.severity, "#555")
            hint = gap.skill_patch_hint[:60] + "…" if len(gap.skill_patch_hint) > 60 else gap.skill_patch_hint
            gap_rows.append(
                f'<TR>'
                f'<TD ALIGN="LEFT" CELLPADDING="2">'
                f'<FONT POINT-SIZE="8" COLOR="{col}"><B>{gtype_icon} {_esc(gap.gap_id)}</B>  '
                f'{_esc(gap.category_tier1)}/{_esc(gap.category_tier2)}</FONT>'
                f'</TD></TR>'
                f'<TR><TD ALIGN="LEFT" CELLPADDING="2">'
                f'<FONT POINT-SIZE="7" COLOR="#444444"><I>{_esc(hint)}</I></FONT>'
                f'</TD></TR>'
            )

        summary_label = (
            '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="1" CELLPADDING="2">'
            '<TR><TD ALIGN="CENTER"><FONT POINT-SIZE="9"><B>Gap Summary</B></FONT></TD></TR>'
            '<TR><TD><FONT POINT-SIZE="1"> </FONT></TD></TR>'
            + "".join(gap_rows)
            + '</TABLE>>'
        )
        with g.subgraph(name="cluster_gap_summary") as gs:
            gs.attr(label="", style="rounded", color="#C0392B", bgcolor="#FFF5F5")
            gs.node("_gap_summary", label=summary_label, shape="plaintext", margin="0")
