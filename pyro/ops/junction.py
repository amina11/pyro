from __future__ import absolute_import, division, print_function


def build_junction_tree(vertices, arcs):
    """
    Builds a Junction Tree from a set of vertices and a set of arcs.

    :param list vertices: a list of vertices.
    :param list arcs: a list of arcs. Each arc is a frozenset of vertices.
    :return: a tuple ``(junctions, edges)`` where ``junctions`` is a list of
        frozensets of vertices, and edges is a list of ``(head,tail)`` pairs
        that index into the junction list. The set of active vertices along an
        edge is thus ``junctions[head] & junction[tail]``.
    """
    raise NotImplementedError
