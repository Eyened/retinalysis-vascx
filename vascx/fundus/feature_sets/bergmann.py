from vascx.fundus.features.base import FeatureSet
from vascx.fundus.features.bifurcations import BifurcationCount
from vascx.fundus.features.caliber import Caliber
from vascx.fundus.features.cre import CRE
from vascx.fundus.features.temporal_angles import TemporalAngle
from vascx.fundus.features.tortuosity import LengthMeasure, Tortuosity
from vascx.fundus.features.vascular_densities import VascularDensity
from vascx.shared.aggregators import median, median_std

bergmann_features = FeatureSet(
    {
        "ta": TemporalAngle(),
        "cre": CRE(),
        "vd": VascularDensity(),
        "diam": Caliber(aggregator=median_std),
        "tort": Tortuosity(length_measure=LengthMeasure.Skeleton, aggregator=median),
        "tort_splines": Tortuosity(
            length_measure=LengthMeasure.Splines, aggregator=median
        ),
        "bif": BifurcationCount(),
    }
)
