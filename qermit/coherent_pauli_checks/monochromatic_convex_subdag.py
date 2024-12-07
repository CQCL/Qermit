from itertools import permutations
from typing import Any

import networkx as nx  # type: ignore


def _subdag_predecessors(
    dag: nx.DiGraph, subdag: int, node_subdag: dict[Any, int]
) -> list[Any]:
    subdag_nodes = _subdag_nodes(subdag=subdag, node_subdag=node_subdag)
    return sum(
        [
            [
                predecessor
                for predecessor in dag.predecessors(node)
                if predecessor not in subdag_nodes
            ]
            for node in subdag_nodes
        ],
        start=[],
    )


def _subdag_successors(
    dag: nx.DiGraph, subdag: int, node_subdag: dict[Any, int]
) -> list[Any]:
    subdag_nodes = _subdag_nodes(subdag=subdag, node_subdag=node_subdag)
    return sum(
        [
            [
                successor
                for successor in dag.successors(node)
                if successor not in subdag_nodes
            ]
            for node in subdag_nodes
        ],
        start=[],
    )


def _subdag_nodes(subdag: int, node_subdag: dict[Any, int]) -> list[Any]:
    return [node for node, s in node_subdag.items() if s == subdag]


def _can_merge(
    dag: nx.DiGraph,
    subdag_one: int,
    subdag_two: int,
    node_subdag: dict[Any, int],
    node_descendants: dict[int, set[int]],
) -> bool:
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


def get_monochromatic_convex_subdag(
    dag: nx.DiGraph, coloured_nodes: set[Any]
) -> dict[Any, int]:
    node_subdag = {node: i for i, node in enumerate(coloured_nodes)}

    node_descendants = {node: nx.descendants(dag, node) for node in dag.nodes}
    for node in node_descendants.keys():
        node_descendants[node].add(node)

    subgraph_merged = True
    while subgraph_merged:
        subgraph_merged = False

        for subdag_one, subdag_two in permutations(node_subdag.values(), 2):
            if subdag_one == subdag_two:
                continue

            if _can_merge(
                dag=dag,
                subdag_one=subdag_one,
                subdag_two=subdag_two,
                node_subdag=node_subdag,
                node_descendants=node_descendants,
            ):
                for node in _subdag_nodes(subdag=subdag_two, node_subdag=node_subdag):
                    node_subdag[node] = subdag_one
                subgraph_merged = True
                break

    return node_subdag
