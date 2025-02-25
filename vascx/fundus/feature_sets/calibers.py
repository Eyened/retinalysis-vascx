# fmt: off
from vascx.fundus.features.caliber import Caliber
from vascx.fundus.features.coverage import Coverage
from vascx.fundus.features.cre import CRE
from vascx.fundus.features.disc_features import DiscFoveaDistance
from vascx.shared.aggregators import median_std
from vascx.shared.features import FeatureSet

full_features = FeatureSet(
    "full",
    {
        # CREs and diameters
        "cre": CRE(),
        "diam": Caliber(aggregator=median_std),

        # general and retina-level features
        "coverage": Coverage(),
        "odfd": DiscFoveaDistance(),
    },
)
