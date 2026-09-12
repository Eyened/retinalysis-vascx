# fmt: off
from rtnls_enface.grids.etdrs import ETDRSRing
from rtnls_enface.grids.specifications import (
    ETDRSGridSpecification,
    GridFieldSpecification,
)

from vascx.faz.features.tortuosity import TortuosityMeasure
from vascx.fundus.features.bifurcation_angles import BifurcationAngles
from vascx.fundus.features.bifurcation_counts import BifurcationCount
from vascx.fundus.features.caliber import Caliber
from vascx.fundus.features.cre import CRE
from vascx.fundus.features.disc_features import DiscFoveaDistance
from vascx.fundus.features.sparsity import Sparsity
from vascx.fundus.features.temporal_angles import TemporalAngle
from vascx.fundus.features.tortuosity import LengthMeasure, Tortuosity, TortuosityMode
from vascx.fundus.features.variance_of_laplacian import VarianceOfLaplacian
from vascx.fundus.features.vascular_densities import VascularDensity
from vascx.shared.aggregators import (
    LengthWeightedAggregator,
    median,
    std,
)
from vascx.shared.features import FeatureSet

ETDRS_FULL_FIELD = GridFieldSpecification(
    grid_spec=ETDRSGridSpecification(
        description="a fovea-centered ETDRS grid with ring radii of 0.5, 1.5, and 3.0 mm"
    ),
    field=ETDRSRing.FullGrid,
)

fs_full = FeatureSet(
    "full",
    [
        # temporal angles
        TemporalAngle(plot=True),

        # CREs and diameters
        CRE(plot=True),
        Caliber(plot=True, aggregator=median),
        Caliber(aggregator=std),

        # vascular densities
        VascularDensity(plot=True),
        VascularDensity(ETDRS_FULL_FIELD),

        # tortuosity on segments
        Tortuosity(plot=True, length_measure=LengthMeasure.Skeleton, aggregator=median),
        Tortuosity(length_measure=LengthMeasure.Splines, aggregator=median),
        Tortuosity(measure=TortuosityMeasure.Curvature, aggregator=median),
        Tortuosity(measure=TortuosityMeasure.Inflections, aggregator=median),

        # normalized tortuosity on segments
        Tortuosity(length_measure=LengthMeasure.Skeleton, aggregator=LengthWeightedAggregator()),
        Tortuosity(length_measure=LengthMeasure.Splines, aggregator=LengthWeightedAggregator()),
        Tortuosity(measure=TortuosityMeasure.Curvature, aggregator=LengthWeightedAggregator()),
        Tortuosity(measure=TortuosityMeasure.Inflections, aggregator=LengthWeightedAggregator()),
        
        # tortuosity on vessels
        Tortuosity(mode=TortuosityMode.Vessels, length_measure=LengthMeasure.Skeleton, aggregator=median),
        Tortuosity(mode=TortuosityMode.Vessels, length_measure=LengthMeasure.Splines, aggregator=median),
        Tortuosity(mode=TortuosityMode.Vessels, measure=TortuosityMeasure.Curvature, aggregator=median),
        Tortuosity(mode=TortuosityMode.Vessels, measure=TortuosityMeasure.Inflections, aggregator=median),

        # bifurcation angles
        BifurcationAngles(plot=True, aggregator=median),
        # Note: we have deprecated BifurcationCount due to low reproducibility scores.
        BifurcationCount(plot=True),

        # general and retina-level features
        Sparsity(plot=True),
        VarianceOfLaplacian(plot=True),
        DiscFoveaDistance(plot=True),
    ],
    description=(
        "Legacy comprehensive fundus biomarker set. Prefer full_v3 for new analyses."
    ),
)
