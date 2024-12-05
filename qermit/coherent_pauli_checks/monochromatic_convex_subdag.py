from itertools import permutations

import networkx as nx  # type: ignore


class MonochromaticConvexSubDAG:
    def __init__(self, dag, node_coloured):
        assert all(dag_node in node_coloured.keys() for dag_node in dag.nodes)
        assert all(dag_node in dag.nodes for dag_node in node_coloured.keys())
        assert nx.is_directed_acyclic_graph(dag)

        self.dag = dag
        self.node_coloured = node_coloured

        self.node_descendants = {
            node: nx.descendants(self.dag, node) for node in self.dag.nodes
        }
        for node in self.node_descendants.keys():
            self.node_descendants[node].add(node)

    def _subdag_nodes(self, subdag, node_subdag):
        return [node for node, s in node_subdag.items() if s == subdag]

    def _subdag_predecessors(self, subdag, node_subdag):
        subdag_nodes = self._subdag_nodes(subdag=subdag, node_subdag=node_subdag)
        return sum(
            [
                [
                    predecessor
                    for predecessor in self.dag.predecessors(node)
                    if predecessor not in subdag_nodes
                ]
                for node in subdag_nodes
            ],
            start=[],
        )

    def _subdag_successors(self, subdag, node_subdag):
        subdag_nodes = self._subdag_nodes(subdag=subdag, node_subdag=node_subdag)
        return sum(
            [
                [
                    successor
                    for successor in self.dag.successors(node)
                    if successor not in subdag_nodes
                ]
                for node in subdag_nodes
            ],
            start=[],
        )

    def _can_merge(self, subdag_one, subdag_two, node_subdag):
        subdag_two_pred = self._subdag_predecessors(
            subdag=subdag_two, node_subdag=node_subdag
        )
        for subdag_one_succ in self._subdag_successors(
            subdag=subdag_one, node_subdag=node_subdag
        ):
            if any(
                descendant in subdag_two_pred
                for descendant in self.node_descendants[subdag_one_succ]
            ):
                return False

        subgraph_one_pred = self._subdag_predecessors(
            subdag=subdag_one, node_subdag=node_subdag
        )
        for subdag_two_succ in self._subdag_successors(
            subdag=subdag_two, node_subdag=node_subdag
        ):
            if any(
                descendant in subgraph_one_pred
                for descendant in self.node_descendants[subdag_two_succ]
            ):
                return False

        return True

    def greedy_merge(self):
        node_subdag = {
            node: i
            for i, node in enumerate(self.node_coloured)
            if self.node_coloured[node]
        }

        subgraph_merged = True
        while subgraph_merged:
            subgraph_merged = False

            for subdag_one, subdag_two in permutations(node_subdag.values(), 2):
                if subdag_one == subdag_two:
                    continue

                if self._can_merge(
                    subdag_one=subdag_one,
                    subdag_two=subdag_two,
                    node_subdag=node_subdag,
                ):
                    for node in self._subdag_nodes(
                        subdag=subdag_two, node_subdag=node_subdag
                    ):
                        node_subdag[node] = subdag_one
                    subgraph_merged = True
                    break

        return node_subdag
