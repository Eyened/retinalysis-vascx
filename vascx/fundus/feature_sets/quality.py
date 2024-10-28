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

bergmann_features = FeatureSet(
    "full",
    {
        "vd": VascularDensity(),
        "vd_total": VascularDensity(MiscField.Total, cut_mask=True),
        "diam": Caliber(aggregator=median_std),

        "tort_segments_skl": Tortuosity(length_measure=LengthMeasure.Skeleton, aggregator=median),
        "tort_segments_spl": Tortuosity(length_measure=LengthMeasure.Splines, aggregator=median),
        "tort_segments_curv": Tortuosity(measure=TortuosityMeasure.Curvature, aggregator=median),
        "tort_segments_infl": Tortuosity(measure=TortuosityMeasure.Inflections, aggregator=median),

        "norm_tort_segments_skl": Tortuosity(length_measure=LengthMeasure.Skeleton, norm_measure=LengthMeasure.Skeleton, aggregator=sum),
        "norm_tort_segments_spl": Tortuosity(length_measure=LengthMeasure.Splines, norm_measure=LengthMeasure.Splines, aggregator=sum),
        "norm_tort_segments_curv": Tortuosity(measure=TortuosityMeasure.Curvature, norm_measure=LengthMeasure.Skeleton, aggregator=sum),
        "norm_tort_segments_infl": Tortuosity(measure=TortuosityMeasure.Inflections, norm_measure=LengthMeasure.Skeleton, aggregator=sum),
        
        "bif_angles": BifurcationAngles(aggregator=mean_median),
        "bif": BifurcationCount(),

        "coverage": Coverage(),
        "lapl": VarianceOfLaplacian()
    },
)
