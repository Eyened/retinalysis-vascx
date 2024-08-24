from vascx.fundus.features.bifurcation_counts import BifurcationCount
from vascx.fundus.features.caliber import Caliber
from vascx.fundus.features.cre import CRE
from vascx.fundus.features.tortuosity import LengthMeasure, Tortuosity
from vascx.shared.aggregators import median, median_std
from vascx.shared.features import FeatureSet

bergmann_features = FeatureSet(
    "bergmann",
    {
        # "ta": TemporalAngle(),
        "cre": CRE(),
        # "vd": VascularDensity(),
        "diam": Caliber(aggregator=median_std),
        "tort": Tortuosity(length_measure=LengthMeasure.Splines, aggregator=median),
        "bif": BifurcationCount(),
    },
)
