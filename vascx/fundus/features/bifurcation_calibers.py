from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from matplotlib import patches

from rtnls_enface.base import Line, Point

from .base import LayerFeature

if TYPE_CHECKING:
    from vascx.fundus.layer import VesselTreeLayer


@dataclass
class BifurcationCalibers(LayerFeature):
    def __init__(self, delta: int = 20):
        """
        Calculation of bifurcation angles.

        Args:
            delta (int): Minimal length for both outgoing segments and for incoming segment, to be considered.
            max_angle (str): Skip bifurcations, with opening angles larger than this value.

        """
        self.delta = delta
        self.max_angle = 160


    def compute_for_bifurcation(self, bif):

        # Skip bifurcations with short outgoing segments
        if bif.outgoing[0].length < self.delta or bif.outgoing[1].length < self.delta:
            return None
        
        # Check for opening angles < max_angle
        # Calculate points and lines for outgoing segments
        to1 = Point(*bif.outgoing[0].spline.get_point_pixels(self.delta))
        to2 = Point(*bif.outgoing[1].spline.get_point_pixels(self.delta))

        line1 = Line(bif.position, to1)
        line2 = Line(bif.position, to2)

        # Calculate and adjust the angle between outgoing segments
        outgoing_angle = line1.counterclockwise_angle_to(line2)
        if outgoing_angle > 180:
            line1, line2 = line2, line1

        outgoing_angle = line1.angle_to(line2)
        if outgoing_angle > self.max_angle:
            return None
        
        outgoing_diam1 = bif.outgoing[0].median_diameter
        outgoing_diam2 = bif.outgoing[1].median_diameter
        if bif.incoming.length < self.delta:
            incoming_diam = None
        else:
            incoming_diam = bif.incoming.median_diameter
        
        # If all three segments are long enough, and the opening angle is smaller than max_angle, return median diameters the segments
        # first two are outgoing, last one is incoming segment's median diameter
        return (outgoing_diam1, outgoing_diam2, incoming_diam)
    
    def compute(self, layer: VesselTreeLayer):
        list_of_diameters = []

        for bif in layer.bifurcations:
            diameters = self.compute_for_bifurcation(bif)
            if diameters is not None:
                list_of_diameters.append(diameters)
        return list_of_diameters
    
    def plot(self, ax, layer: VesselTreeLayer, **kwargs):
        pass