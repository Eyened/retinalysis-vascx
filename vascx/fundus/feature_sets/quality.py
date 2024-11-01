from rtnls_enface.grids.etdrs import MiscField
from vascx.faz.features.tortuosity import TortuosityMeasure
from vascx.fundus.features.bifurcation_angles import BifurcationAngles
from vascx.fundus.features.bifurcation_counts import BifurcationCount
from vascx.fundus.features.caliber import Caliber
from vascx.fundus.features.coverage import Coverage
from vascx.fundus.features.cre import CRE
from vascx.fundus.features.temporal_angles import TemporalAngle
from vascx.fundus.features.tortuosity import LengthMeasure, Tortuosity, TortuosityMode
from vascx.fundus.features.variance_of_laplacian import VarianceOfLaplacian
from vascx.fundus.features.vascular_densities import VascularDensity
from vascx.shared import aggregators
from vascx.shared.aggregators import mean_median, median, median_std, sum
from vascx.shared.features import FeatureSet

# Created for evaluation of image quality and its effect on some features

quality_features = FeatureSet(
    "quality",
    {
        "coverage": Coverage(),
        "lapl": VarianceOfLaplacian()
    },
)
