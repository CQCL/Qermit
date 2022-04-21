"""Visualise TaskGraph using graphviz."""
from typing import Iterable, Optional, Tuple, TYPE_CHECKING
import graphviz as gv  # type: ignore

from qermit.taskgraph.mittask import IOTask

if TYPE_CHECKING:
    from qermit.taskgraph.mittask import MitTask
    import networkx as nx  # type: ignore

# qermit colours
_COLOURS = {
    "background": "white",
    "node": "#c0dd8e",
    "edge": "#6bb24d",
    "dark": "black",
    "node_border": "white",
    "port_border": "#6bb24d",
}
_FONTFACE = "monospace"

_HTML_LABEL_TEMPLATE = """
<TABLE BORDER="{border_width}" CELLBORDER="0" CELLSPACING="1" CELLPADDING="1" BGCOLOR="{node_back_color}" COLOR="{border_colour}">

{inputs_row}

    <TR>
        <TD>
            <TABLE BORDER="0" CELLBORDER="0">
                <TR>
                    <TD><FONT POINT-SIZE="11.0" FACE="{fontface}" COLOR="{label_color}"><B>{node_label}</B></FONT></TD>
                </TR>
            </TABLE>
        </TD>
    </TR>

{outputs_row}

</TABLE>
"""


def _format_html_label(**kwargs):
    _HTML_LABEL_DEFAULTS = {
        "label_color": _COLOURS["dark"],
        "node_back_color": _COLOURS["node"],
        "inputs_row": "",
        "outputs_row": "",
        "border_colour": _COLOURS["port_border"],
        "border_width": "1",
        "fontface": _FONTFACE,
    }
    return _HTML_LABEL_TEMPLATE.format(**{**_HTML_LABEL_DEFAULTS, **kwargs})


_HTML_PORTS_ROW_TEMPLATE = """
    <TR>
        <TD>
            <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="3" CELLPADDING="2">
                <TR>
                    {port_cells}
                </TR>
            </TABLE>
        </TD>
    </TR>
"""

_HTML_PORT_TEMPLATE = (
    '<TD BGCOLOR="{back_colour}" COLOR="{border_colour}"'
    ' PORT="{port}" BORDER="{border_width}">'
    '<FONT POINT-SIZE="10.0" FACE="{fontface}" COLOR="{font_colour}">{port_name}</FONT></TD>'
)


def _html_ports(ports: Iterable[str]) -> str:

    return _HTML_PORTS_ROW_TEMPLATE.format(
        port_cells="".join(
            _HTML_PORT_TEMPLATE.format(
                port_name=port.replace("in_", "").replace("out_", ""),
                port=port,
                back_colour=_COLOURS["background"],
                font_colour=_COLOURS["dark"],
                border_width="1",
                border_colour=_COLOURS["port_border"],
                fontface=_FONTFACE,
            )
            for port in ports
        )
    )


def _node_features(node_name: str, node: "MitTask") -> Tuple[str, str]:
    """Calculate node label (first) and colour (second)."""

    fillcolor = _COLOURS["node"]
    node_label = node_name

    if isinstance(node, IOTask):
        fillcolor = _COLOURS["background"]

    return node_label, fillcolor


def _convert_nodename(node: "MitTask") -> str:
    if node == IOTask.Input:
        return "Inputs"
    if node == IOTask.Output:
        return "Outputs"
    return node._label.replace("::", "-").replace("<", "").replace(">", "")


def _taskgraph_to_graphviz(
    nx_graph: "nx.MultiDiGraph",
    initial_graph: Optional[gv.Digraph] = None,
    label: str = "MitRes",
    prefix: str = "",
) -> gv.Digraph:
    gv_graph = (
        gv.Digraph(
            label,
            strict=False,  # stops multiple shared edges being merged
        )
        if initial_graph is None
        else initial_graph
    )
    graph_atrr = {
        "rankdir": "",
        "ranksep": "0.1",
        "nodesep": "0.15",
        "margin": "0",
        "bgcolor": _COLOURS["background"],
    }
    gv_graph.attr(**graph_atrr)

    for node in nx_graph.nodes:
        node_identifier = prefix + _convert_nodename(node)
        node_name = node_identifier

        node_label, fillcolor = _node_features(node_name, node)

        # node is a table
        # first row is a single cell containing a single row table of inputs
        # second row is table containing single cell of node_label
        # third row is single cell containing a single row table of outputs

        in_ports = [f"in_{i}" for i in range(nx_graph.in_degree(node))]
        out_ports = [f"out_{i}" for i in range(nx_graph.out_degree(node))]

        html_label = _format_html_label(
            node_back_color=fillcolor,
            node_label=node_label,
            inputs_row=_html_ports(in_ports) if in_ports else "",
            outputs_row=_html_ports(out_ports) if out_ports else "",
            border_colour=_COLOURS["background"]
            if fillcolor == _COLOURS["background"]
            else _COLOURS["node_border"],
        )
        gv_graph.node(
            node_identifier,
            label=f"<{html_label}>",
            shape="plain",
        )

    edge_attr = {
        "penwidth": "1.5",
        "arrowhead": "none",
        "arrowsize": "1.0",
        "fontname": _FONTFACE,
        "fontsize": "9",
        "color": _COLOURS["edge"],
        "fontcolor": "black",
    }
    for edge in nx_graph.edges:
        src_node, tgt_node, _ = edge
        src_port = edge[2][0]
        tgt_port = edge[2][1]

        src_nodename = f"{_convert_nodename(src_node)}:out_{src_port}"
        tgt_nodename = f"{_convert_nodename(tgt_node)}:in_{tgt_port}"
        gv_graph.edge(
            src_nodename,
            tgt_nodename,
            **edge_attr,
        )

    return gv_graph
