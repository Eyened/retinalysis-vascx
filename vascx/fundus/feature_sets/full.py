# fmt: off
from rtnls_enface.grids.etdrs import MiscField
from vascx.faz.features.tortuosity import TortuosityMeasure
from vascx.fundus.features.bifurcation_angles import BifurcationAngles
from vascx.fundus.features.bifurcation_counts import BifurcationCount
from vascx.fundus.features.caliber import Caliber
from vascx.fundus.features.coverage import Coverage
from vascx.fundus.features.cre import CRE
from vascx.fundus.features.disc_features import DiscFoveaDistance
from vascx.fundus.features.temporal_angles import TemporalAngle
from vascx.fundus.features.tortuosity import LengthMeasure, Tortuosity, TortuosityMode
from vascx.fundus.features.variance_of_laplacian import VarianceOfLaplacian
from vascx.fundus.features.vascular_densities import VascularDensity
from vascx.shared.aggregators import mean_median, median, median_std, sum
from vascx.shared.features import FeatureSet

full_features = FeatureSet(
    "full",
    {
        # temporal angles
        "ta": TemporalAngle(),

        # CREs and diameters
        "cre": CRE(),
        "diam": Caliber(aggregator=median_std),

        # vascular densities
        "vd": VascularDensity(),
        "vd_total": VascularDensity(MiscField.Total, cut_mask=True),

        # tortuosity on segments
        "tort_segments_skl": Tortuosity(length_measure=LengthMeasure.Skeleton, aggregator=median),
        "tort_segments_spl": Tortuosity(length_measure=LengthMeasure.Splines, aggregator=median),
        "tort_segments_curv": Tortuosity(measure=TortuosityMeasure.Curvature, aggregator=median),
        "tort_segments_infl": Tortuosity(measure=TortuosityMeasure.Inflections, aggregator=median),

        # normalized tortuosity on segments
        "norm_tort_segments_skl": Tortuosity(length_measure=LengthMeasure.Skeleton, norm_measure=LengthMeasure.Skeleton, aggregator=sum),
        "norm_tort_segments_spl": Tortuosity(length_measure=LengthMeasure.Splines, norm_measure=LengthMeasure.Splines, aggregator=sum),
        "norm_tort_segments_curv": Tortuosity(measure=TortuosityMeasure.Curvature, norm_measure=LengthMeasure.Skeleton, aggregator=sum),
        "norm_tort_segments_infl": Tortuosity(measure=TortuosityMeasure.Inflections, norm_measure=LengthMeasure.Skeleton, aggregator=sum),
        
        # tortuosity on vessels
        "tort_vessels_skl": Tortuosity(mode=TortuosityMode.Vessels, length_measure=LengthMeasure.Skeleton, aggregator=median),
        "tort_vessels_spl": Tortuosity(mode=TortuosityMode.Vessels, length_measure=LengthMeasure.Splines, aggregator=median),
        "tort_vessels_curv": Tortuosity(mode=TortuosityMode.Vessels, measure=TortuosityMeasure.Curvature, aggregator=median),
        "tort_vessels_infl": Tortuosity(mode=TortuosityMode.Vessels, measure=TortuosityMeasure.Inflections, aggregator=median),

        # bifurcation angles        
        "bif_angles": BifurcationAngles(aggregator=mean_median),
        "bif": BifurcationCount(),

        # general and retina-level features
        "coverage": Coverage(),
        "lapl": VarianceOfLaplacian(),
        "odfd": DiscFoveaDistance(),
    },
)
