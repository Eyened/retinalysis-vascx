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
        TemporalAngle(plot=True),
        CRE(plot=True),
        VascularDensity(plot=True),
        Caliber(plot=True, aggregator=median),
        Caliber(aggregator=std),
        Tortuosity(plot=True, length_measure=LengthMeasure.Skeleton, aggregator=median),
    ],
    description=(
        "Compact set aligned with Bergmann et al. analyses: temporal angle, CRE, "
        "vascular density, caliber, and tortuosity."
    ),
)
