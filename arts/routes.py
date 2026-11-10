"""Legal road-segment routes, shortest-path distances, and diverse OD route libraries."""
from heapq import heapify, heappop, heappush
import networkx as nx


def distance(path, data):
    return sum(data.roads[e][0] for e in path)


def valid(path, q, data):
    if not path or q not in data.endpoints: return False
    starts, ends = data.endpoints[q]
    return (path[0] in starts and path[-1] in ends and
            all(e in data.roads for e in path) and
            all(b in data.successors[a] for a, b in zip(path, path[1:])))


def overlap(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a or b else 1.


def shortest_distance(q, data):
    """Compute the minimum feasible OD path length on the complete public road network."""
    if q in data.shortest_lengths: return data.shortest_lengths[q]
    starts, ends = data.endpoints[q]
    heap = [(data.roads[e][0], e) for e in sorted(starts)]
    best = {e: d for d, e in heap}
    heapify(heap)
    while heap:
        d, u = heappop(heap)
        if d > best[u]: continue
        if u in ends:
            data.shortest_lengths[q] = d
            return d
        for v in sorted(data.successors[u]):
            nxt = d + data.roads[v][0]
            if nxt < best.get(v, float('inf')):
                best[v] = nxt
                heappush(heap, (nxt, v))
    raise ValueError(f'No feasible road-network path for OD {q}')


def validate_library(data, max_detour=None, max_routes=10, max_overlap=0.8):
    """Check endpoint, connectivity, shortest-route, detour, and diversity constraints."""
    for q, paths in data.library.items():
        if q not in data.endpoints or not paths or len(paths) > max_routes:
            raise ValueError(f'Invalid OD route-library entry: {q}')
        if any(not valid(path, q, data) or len(path) != len(set(path)) for path in paths):
            raise ValueError(f'Invalid road-network route for OD {q}')
        if len(paths) != len(set(paths)):
            raise ValueError(f'Duplicate candidate route for OD {q}')
        shortest = shortest_distance(q, data)
        if abs(min(distance(path, data) for path in paths) - shortest) > 1e-7 * max(1., shortest):
            raise ValueError(f'Shortest OD route is missing from the library: {q}')
        if max_detour is not None and any(distance(path, data) > max_detour * shortest + 1e-8 for path in paths):
            raise ValueError(f'Detour limit exceeded for OD {q}')
        if any(overlap(path, prev) > max_overlap + 1e-12
               for i, path in enumerate(paths) for prev in paths[:i]):
            raise ValueError(f'Route diversity threshold exceeded for OD {q}')
    return True


def build_library(data, k=10, max_detour=3., max_overlap=0.8):
    """Enumerate loopless OD routes by Yen's algorithm with detour and diversity limits."""
    if k < 1 or max_detour < 1 or not 0 <= max_overlap <= 1:
        raise ValueError('Invalid route-library parameters')
    graph = nx.DiGraph()
    graph.add_nodes_from(sorted(data.roads))
    for u in sorted(data.successors):
        for v in sorted(data.successors[u]):
            graph.add_edge(u, v, weight=data.roads[v][0])
    result = {}
    data.shortest_lengths.clear()
    for q in sorted(data.endpoints):
        starts, ends = data.endpoints[q]
        src, dst = ('source', q), ('target', q)
        graph.add_nodes_from((src, dst))
        for e in sorted(starts): graph.add_edge(src, e, weight=data.roads[e][0])
        for e in sorted(ends): graph.add_edge(e, dst, weight=0.)
        chosen, shortest = [], None
        try:
            for route in nx.shortest_simple_paths(graph, src, dst, weight='weight'):
                path = tuple(route[1:-1])
                length = distance(path, data)
                if shortest is None: shortest = length
                if length > max_detour * shortest + 1e-8: break
                if all(overlap(path, prev) <= max_overlap for prev in chosen): chosen.append(path)
                if len(chosen) == k: break
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        finally:
            graph.remove_nodes_from((src, dst))
        if chosen:
            result[q] = chosen
            data.shortest_lengths[q] = shortest
    data.library = result
    validate_library(data, max_detour, k, max_overlap)
    return result
