from vascx.faz.features.bifurcations import BifurcationCount
from vascx.faz.features.caliber import Caliber
from vascx.faz.features.faz_params import FazParameter, FazParameterType
from vascx.faz.features.tortuosity import LengthMeasure, Tortuosity
from vascx.faz.features.vascular_densities import VascularDensity
from vascx.shared.aggregators import median, median_std
from vascx.shared.features import FeatureSet

from rtnls_enface.grids.etdrs import Quadrant, Ring

basic_features = FeatureSet(
    "basic",
    {
        "vd": VascularDensity(),
        "vd_inner": VascularDensity(Ring.Inner),
        "vd_outer": VascularDensity(Ring.Outer),
        "vd_superior": VascularDensity(Quadrant.Superior),
        "vd_inferior": VascularDensity(Quadrant.Inferior),
        "vd_left": VascularDensity(Quadrant.Left),
        "vd_right": VascularDensity(Quadrant.Right),
        "faz_perimeter": FazParameter(FazParameterType.PerimeterLength),
        "faz_area": FazParameter(FazParameterType.Area),
        "diam": Caliber(min_numpoints=5, aggregator=median_std),
        "tort": Tortuosity(
            length_measure=LengthMeasure.Splines, min_numpoints=5, aggregator=median
        ),
        "bif": BifurcationCount(),
    },
)
