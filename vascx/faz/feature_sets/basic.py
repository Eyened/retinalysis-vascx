from vascx.faz.features.bifurcations import BifurcationCount
from vascx.faz.features.caliber import Caliber
from vascx.faz.features.tortuosity import LengthMeasure, Tortuosity
from vascx.faz.features.vascular_densities import VascularDensity
from vascx.shared.aggregators import median, median_std
from vascx.shared.features import FeatureSet

basic_features = FeatureSet(
    "basic",
    {
        "vd": VascularDensity(),
        "diam": Caliber(min_numpoints=5, aggregator=median_std),
        "tort": Tortuosity(
            length_measure=LengthMeasure.Splines, min_numpoints=5, aggregator=median
        ),
        "bif": BifurcationCount(),
    },
)
