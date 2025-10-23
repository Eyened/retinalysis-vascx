from vascx.faz.features.bifurcations import BifurcationCount
from vascx.faz.features.caliber import Caliber
from vascx.faz.features.faz_params import FazParameter, FazParameterType
from vascx.faz.features.tortuosity import LengthMeasure, Tortuosity
from vascx.faz.features.vascular_densities import VascularDensity
from vascx.shared.aggregators import median, median_std
from vascx.shared.features import FeatureSet

from rtnls_enface.grids.etdrs import ETDRSQuadrant, ETDRSRing

basic_features = FeatureSet(
    "basic",
    {
        "vd": VascularDensity(),
        "vd_inner": VascularDensity(ETDRSRing.Inner),
        "vd_outer": VascularDensity(ETDRSRing.Outer),
        "vd_superior": VascularDensity(ETDRSQuadrant.Superior),
        "vd_inferior": VascularDensity(ETDRSQuadrant.Inferior),
        "vd_left": VascularDensity(ETDRSQuadrant.Left),
        "vd_right": VascularDensity(ETDRSQuadrant.Right),
        "faz_perimeter": FazParameter(FazParameterType.PerimeterLength),
        "faz_area": FazParameter(FazParameterType.Area),
        "diam": Caliber(min_numpoints=5, aggregator=median_std),
        "tort": Tortuosity(
            length_measure=LengthMeasure.Splines, min_numpoints=5, aggregator=median
        ),
        "bif": BifurcationCount(),
    },
)
