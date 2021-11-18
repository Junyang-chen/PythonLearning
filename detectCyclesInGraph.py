"""
Determine if a graph is DAG(Directed Acyclic Graph)
"""
from collections import OrderedDict

def detect_cycle(inputs):
    if not inputs:
        return True
    graph = OrderedDict()
    for node_edge in inputs:
        node = node_edge[0]
        edges = node_edge[1]
        edge_set = graph.get(node, set())
        edge_set.update(edges)
        graph[node] = edge_set

    def detect_cycle_helper(node):
        visited.add(node)
        stack.add(node)
        for neighbour in graph.get(node, []):
            if neighbour not in visited:
                return detect_cycle_helper(neighbour)
            elif neighbour in stack:
                return True
        stack.remove(node)
        return False

    visited = set()
    stack = set()
    for node, edges in graph.items():
        if node not in visited:
            if detect_cycle_helper(node):
                return True
    return False







testcase1 = [[[0, [1, 2]],
              [1, [2, 3]],
              [2, [3]]], False]

testcase2 = [[[0, [1, 2]],
              [1, [2, 3]],
              [2, [3]],
               [3, [3]]], True]

for testcase in [testcase1, testcase2]:
    assert detect_cycle(testcase[0]) == testcase[1]
