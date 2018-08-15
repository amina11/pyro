from __future__ import absolute_import, division, print_function

import pytest

from pyro.ops.junction import build_junction_tree

JUNCTION_TREE_EXAMPLES = [
    ([0], []),
    ([0], [[0]]),
    ([0], [[0], [0]]),
    ([0, 1], [[0], [1]]),
    ([0, 1], [[0], [1], [0, 1]]),
    ([0, 1, 2], [[0, 1], [1, 2], [2, 0]]),
    ([0, 1, 2], [[0], [0, 1], [1, 2]]),
    ([0, 1, 2], [[0], [0, 1], [0, 1, 2]]),
    # Example (i) from pp. 50 of Cowell, Dawid, Lauritzen, Spiegelhalter (1999):
    ([0, 1, 2, 3, 4, 5], [[0, 1], [0, 1, 2], [1, 2, 3], [3, 4], [3, 4, 5]]),
]


@pytest.mark.xfail(reason='TODO(fritzo)')
@pytest.mark.parametrize('vertices,arcs', JUNCTION_TREE_EXAMPLES)
def test_build_junction_tree(vertices, arcs):
    arcs = [frozenset(e) for e in arcs]

    junctions, edges = build_junction_tree(vertices, arcs)

    assert isinstance(junctions, list)
    assert isinstance(edges, list)
    for junction in junctions:
        assert isinstance(junction, frozenset)
        assert all(v in vertices for v in junction)
    for head, tail in arcs:
        intersection = junctions[head] & junctions[tail]
        assert intersection
        assert intersection != junctions[head]
        assert intersection != junctions[tail]
    for arc in arcs:
        assert any(arc in junction for junction in junctions), arc
