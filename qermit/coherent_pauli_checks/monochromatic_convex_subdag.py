from itertools import combinations
from typing import Any

import networkx as nx  # type: ignore


def _subdag_nodes(subdag: int, node_subdag: dict[Any, int]) -> list[Any]:
    """Get all nodes in a given sub-DAG.

    :param subdag: The sub-DAG whose nodes should be retrieved.
    :param node_subdag: Map from node to the sub-DAG to which the
        node belongs.
    :return: List of nodes belonging to given sub-DAG.
    """
    return [node for node, s in node_subdag.items() if s == subdag]


def _subdag_predecessors(
    dag: nx.DiGraph, subdag: int, node_subdag: dict[Any, int]
) -> list[Any]:
    """Retrieve all nodes not in given sub-DAG with successors in sub-DAG.

    :param dag: Directed Acyclic Graph.
    :param subdag: Sub-DAG to retrieve predecessors of.
    :param node_subdag: Map from node to the sub-DAG to which it belongs.
    :return: Nodes with successors in given sub-DAG.
    """
    subdag_nodes = _subdag_nodes(subdag=subdag, node_subdag=node_subdag)
    return sum(
        [
            [
                predecessor
                for predecessor in dag.predecessors(node)
                # Exclude nodes in subdag.
                if predecessor not in subdag_nodes
            ]
            for node in subdag_nodes
        ],
        start=[],
    )


def _subdag_successors(
    dag: nx.DiGraph, subdag: int, node_subdag: dict[Any, int]
) -> list[Any]:
    """Retrieve all nodes not in given sub-DAG with predecessors in sub-DAG.

    :param dag: Directed Acyclic Graph.
    :param subdag: Sub-DAG to retrieve successors of.
    :param node_subdag: Map from node to the sub-DAG to which it belongs.
    :return: Nodes with predecessors in given sub-DAG.
    """
    subdag_nodes = _subdag_nodes(subdag=subdag, node_subdag=node_subdag)
    return sum(
        [
            [
                successor
                for successor in dag.successors(node)
                # Exclude nodes in subdag.
                if successor not in subdag_nodes
            ]
            for node in subdag_nodes
        ],
        start=[],
    )


def get_monochromatic_convex_subdag(
    dag: nx.DiGraph, coloured_nodes: list[Any]
) -> dict[Any, int]:
    """Retrieve assignment of coloured nodes to sub-DAGs.
    The assignment aims to minimise the number of sub-DAGs.

    :param dag: Directed Acyclic Graph.
    :param coloured_nodes: The nodes which are coloured.
    :return: Map from node to the sub-DAG to which it belongs.
    """

    node_descendants = {node: nx.descendants(dag, node) for node in dag.nodes}
    for node in node_descendants.keys():
        node_descendants[node].add(node)

    def _can_merge(
        dag: nx.DiGraph,
        subdag_one: int,
        subdag_two: int,
        node_subdag: dict[Any, int],
    ) -> bool:
        """Determine if two sub-DAGs can be merged. This will be the case if
        there are no paths between predecessors of one sub-DAG to successors of
        the other.

        :param dag: Directed Acyclic Graph.
        :param subdag_one: First sub-DAG.
        :param subdag_two: Second sub-DAG.
        :param node_subdag: Map from node to the sub-DAG to which it belongs.
        :return: Boolean value indicating if the two sub-DAGs can be merged.
        """

        subdag_two_pred = _subdag_predecessors(
            dag=dag, subdag=subdag_two, node_subdag=node_subdag
        )
        for subdag_one_succ in _subdag_successors(
            dag=dag, subdag=subdag_one, node_subdag=node_subdag
        ):
            if any(
                descendant in subdag_two_pred
                for descendant in node_descendants[subdag_one_succ]
            ):
                return False

        subgraph_one_pred = _subdag_predecessors(
            dag=dag, subdag=subdag_one, node_subdag=node_subdag
        )
        for subdag_two_succ in _subdag_successors(
            dag=dag, subdag=subdag_two, node_subdag=node_subdag
        ):
            if any(
                descendant in subgraph_one_pred
                for descendant in node_descendants[subdag_two_succ]
            ):
                return False

        return True

    node_subdag = {node: i for i, node in enumerate(coloured_nodes)}

    subgraph_merged = True
    while subgraph_merged:
        subgraph_merged = False

        # Try to merge all pairs of sub-DAGs
        for subdag_one, subdag_two in combinations(set(node_subdag.values()), 2):
            if _can_merge(
                dag=dag,
                subdag_one=subdag_one,
                subdag_two=subdag_two,
                node_subdag=node_subdag,
            ):
                for node in _subdag_nodes(subdag=subdag_two, node_subdag=node_subdag):
                    node_subdag[node] = subdag_one
                subgraph_merged = True
                break

    return node_subdag
