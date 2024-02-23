from typing import List
from vascx.segment import merge_segments

from vessel_analysis import Segment


class VesselResolver:
    pass


class RecursiveWeightedAverageResolver(VesselResolver):
    def __init__(self, prop_name="median_diameter"):
        super().__init__()
        self.prop_name = prop_name

        self.prop_values = {}
        self.visited = None

    def _set_prop_value(self, seg, prop_name, value):
        if seg not in self.prop_values:
            self.prop_values[seg] = {}
        self.prop_values[seg][prop_name] = value

    def _get_prop_value(self, seg, prop_name):
        return self.prop_values[seg][prop_name]

    def _mark_as_visited(self, seg):
        self.visited.add(seg)

    def _is_visited(self, seg):
        return seg in self.visited

    def _recursive_set_prop_values(self, seg, ignore=None):
        if ignore is None:
            ignore = set()

        self._mark_as_visited(seg)
        neighbors = set(seg.neighbors) - ignore
        neighbors = [n for n in neighbors if not self._is_visited(n)]

        median_diameter = seg.median_diameter if seg.median_diameter is not None else 0

        if len(neighbors) == 0:
            self._set_prop_value(seg, "agg_value", median_diameter)
            self._set_prop_value(seg, "agg_length", seg.length)
            return median_diameter, seg.length
        else:
            tmp = [
                self._recursive_set_prop_values(n, ignore=set(seg.neighbors))
                for n in neighbors
            ]

            tmp = sorted(tmp, key=lambda x: x[0], reverse=True)  # sort by mean diam
            max_agg_diameter, agg_length = tmp[0]
            agg_value = (
                max_agg_diameter * agg_length + median_diameter * seg.length
            ) / (agg_length + seg.length)
            self._set_prop_value(seg, "agg_value", agg_value)
            self._set_prop_value(seg, "agg_length", agg_length + seg.length)
            return agg_value, agg_length

    def _recursive_merge(self, seg, ignore=None) -> (List[Segment], List[Segment]):
        if ignore is None:
            ignore = set()

        self._mark_as_visited(seg)

        connector = seg.end
        neighbors = set([n for n in seg.neighbors if not self._is_visited(n)]) - ignore

        for n in neighbors:
            if n.end == connector:
                n.reverse()

        # hack: remove spurious stumps
        neighbors = [
            n
            for n in neighbors
            if not (
                len(set(n.neighbors) - set(seg.neighbors)) == 1
                and n.length < 10
                and n.median_diameter > seg.median_diameter
            )
        ]

        open_segment, completed_segments = [seg], []
        if len(neighbors) > 0:
            neighbors = sorted(
                neighbors,
                key=lambda x: self._get_prop_value(x, "agg_value"),
                reverse=True,
            )
            os, cs = self._recursive_merge(neighbors[0], ignore=set(seg.neighbors))

            open_segment += os
            completed_segments += cs

            for n in neighbors[1:]:
                os, cs = self._recursive_merge(n, ignore=set(seg.neighbors))

                completed_segments.append(merge_segments(*os))
                completed_segments += cs
        return open_segment, completed_segments

    def __call__(self, segments):
        self.visited = set()

        for seg in segments:
            self._recursive_set_prop_values(seg)

        self.visited = set()
        vessels = []
        for seg in segments:
            open_segment, completed_segments = self._recursive_merge(seg)
            vessels += completed_segments
            vessels.append(merge_segments(*open_segment))

        for i, seg in enumerate(vessels):
            seg.index = i

        return vessels
