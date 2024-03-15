from vascx.analysis.aggregators import median, median_std
from vascx.features.base import FeatureSet
from vascx.features.bifurcations import BifurcationCount
from vascx.features.caliber import Caliber
from vascx.features.cre import CRE
from vascx.features.temporal_angles import TemporalAngle
from vascx.features.tortuosity import Tortuosity
from vascx.features.vascular_densities import VascularDensity

bergmann_features = FeatureSet(
    {
        "ta": TemporalAngle(),
        "cre": CRE(),
        "vd": VascularDensity(),
        "diam": Caliber(aggregator=median_std),
        "tort": Tortuosity(aggregator=median),
        "bif": BifurcationCount(),
    }
)
