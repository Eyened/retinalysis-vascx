from vascx.fundus.features.bifurcation_counts import BifurcationCount
from vascx.fundus.features.caliber import Caliber
from vascx.fundus.features.cre import CRE
from vascx.fundus.features.temporal_angles import TemporalAngle
from vascx.fundus.features.tortuosity import LengthMeasure, Tortuosity
from vascx.fundus.features.vascular_densities import VascularDensity
from vascx.shared.aggregators import median, std
from vascx.shared.features import FeatureSet

fs_bergmann = FeatureSet(
    "bergmann",
    [
        TemporalAngle(),
        CRE(),
        VascularDensity(),
        Caliber(aggregator=median),
        Caliber(aggregator=std),
        Tortuosity(length_measure=LengthMeasure.Skeleton, aggregator=median),
        # Note: we have deprecated BifurcationCount due to low reproducibility scores.
        BifurcationCount(),
    ],
    description=(
        "Compact set aligned with Bergmann et al. analyses: temporal angle, CRE, "
        "vascular density, caliber, tortuosity, and bifurcation count."
    ),
)
