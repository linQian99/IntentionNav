"""Keep object termination consistent with the endpoint of its actual map route."""
from __future__ import annotations

import math


def plan_object_endpoint(walkable_map, current, predicted_center) -> dict | None:
    """Remember the same projected endpoint used by shortest_path_2d.

    A learned object's center may be occupied. The existing path planner snaps
    it to a walkable cell within its existing radius. Do not enlarge that radius
    or treat failure to find a connected route as arrival. Inputs contain only
    the model prediction, agent pose and permitted nonsemantic map.
    """
    center = tuple(float(v) for v in predicted_center[:2])
    if len(center) != 2 or not all(math.isfinite(v) for v in center):
        raise ValueError('Invalid learned object center')
    path = walkable_map.shortest_path_2d(tuple(current[:2]), center)
    if not path:
        return None
    endpoint = tuple(float(v) for v in path[-1])
    if not walkable_map.is_walkable(*endpoint):
        raise ValueError('Path endpoint is not navigable')
    return {'predicted_center_xy': list(center), 'navigation_endpoint_xy': list(endpoint),
            'center_is_walkable': bool(walkable_map.is_walkable(*center)),
            'projection_distance_m': math.dist(center, endpoint),
            'source': 'existing_shortest_path_2d_endpoint',
            'arrival_tolerance_m': 1e-5}


def reached_projected_endpoint(walkable_map, current, plan: dict | None) -> bool:
    """Only an actual arrival at a projected object endpoint can terminate."""
    if plan is None or plan['center_is_walkable']:
        return False
    endpoint = plan['navigation_endpoint_xy']
    return bool(walkable_map.is_walkable(*endpoint)
                and math.dist(current[:2], endpoint) <= 1e-5)
