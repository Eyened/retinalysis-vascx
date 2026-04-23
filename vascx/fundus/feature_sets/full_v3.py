# fmt: off
from rtnls_enface.grids.disc_centered import DiscCenteredRing
from rtnls_enface.grids.ellipse import EllipseField
from rtnls_enface.grids.etdrs import ETDRSRing
from rtnls_enface.grids.hemifields import HemifieldField
from rtnls_enface.grids.specifications import (
    DiscCenteredGridSpecification,
    EllipseGridSpecification,
    ETDRSGridSpecification,
    GridFieldSpecification,
    HemifieldGridSpecification,
)

from vascx.fundus.features.bifurcation_angles import BifurcationAngles
from vascx.fundus.features.caliber import Caliber
from vascx.fundus.features.cre import CRE, CREMode
from vascx.fundus.features.disc_features import DiscFoveaDistance
from vascx.fundus.features.sparsity import Sparsity, SparsityMode
from vascx.fundus.features.temporal_angles import TemporalAngle
from vascx.fundus.features.tortuosity import (
    LengthMeasure,
    Tortuosity,
    TortuosityMeasure,
    TortuosityMode,
)
from vascx.fundus.features.vascular_densities import VascularDensity
from vascx.shared.aggregators import LengthWeightedAggregator, mean, median
from vascx.shared.features import FeatureSet

HMF_SUP = GridFieldSpecification(HemifieldGridSpecification(), HemifieldField.Superior)
HMF_INF = GridFieldSpecification(HemifieldGridSpecification(), HemifieldField.Inferior)
DISC_FULL = GridFieldSpecification(DiscCenteredGridSpecification(), DiscCenteredRing.FullGrid)
ELLIPSE_FULL = GridFieldSpecification(EllipseGridSpecification(), EllipseField.FullGrid)
ETDRS_FULL = GridFieldSpecification(ETDRSGridSpecification(), ETDRSRing.FullGrid)

fs_full_v3 = FeatureSet(
    "full_v3",
    [
        TemporalAngle(),

        # bifurcation angles (full, superior, inferior)
        BifurcationAngles(aggregator=mean),
        BifurcationAngles(grid_field=HMF_SUP, aggregator=mean),
        BifurcationAngles(grid_field=HMF_INF, aggregator=mean),

        # caliber (full, superior, inferior)
        Caliber(aggregator=median),
        Caliber(grid_field=HMF_SUP, aggregator=median),
        Caliber(grid_field=HMF_INF, aggregator=median),

        # caliber (length-weighted)
        Caliber(aggregator=LengthWeightedAggregator()),
        Caliber(grid_field=HMF_SUP, aggregator=LengthWeightedAggregator()),
        Caliber(grid_field=HMF_INF, aggregator=LengthWeightedAggregator()),
        Caliber(grid_field=DISC_FULL, aggregator=LengthWeightedAggregator()),

        # CRE: temporal variants in sup/inf/full; nasal and full variants on full grid
        CRE(CREMode.Temporal),
        CRE(CREMode.Temporal, hemifield=HemifieldField.Superior),
        CRE(CREMode.Temporal, hemifield=HemifieldField.Inferior),
        CRE(CREMode.Nasal),
        CRE(CREMode.Full),

        # tortuosity (segments) — Distance and Curvature
        # whole image (non-normalized median)
        Tortuosity(
            mode=TortuosityMode.Segments,
            measure=TortuosityMeasure.Distance,
            length_measure=LengthMeasure.Splines,
            aggregator=median,
        ),
        Tortuosity(
            mode=TortuosityMode.Segments,
            measure=TortuosityMeasure.Curvature,
            length_measure=LengthMeasure.Splines,
            aggregator=median,
        ),
        # vessels tortuosity
        Tortuosity(mode=TortuosityMode.Vessels, aggregator=LengthWeightedAggregator()),
        # whole image (length-weighted normalized)
        Tortuosity(
            mode=TortuosityMode.Segments,
            measure=TortuosityMeasure.Distance,
            length_measure=LengthMeasure.Splines,
            aggregator=LengthWeightedAggregator(),
        ),
        Tortuosity(
            mode=TortuosityMode.Segments,
            max_segment_len=0.2,
            measure=TortuosityMeasure.Distance,
            length_measure=LengthMeasure.Splines,
            aggregator=LengthWeightedAggregator(),
        ),
        Tortuosity(
            mode=TortuosityMode.Segments,
            measure=TortuosityMeasure.Curvature,
            length_measure=LengthMeasure.Splines,
            aggregator=LengthWeightedAggregator(),
        ),
        # ETDRS total (length-weighted normalized)
        Tortuosity(
            mode=TortuosityMode.Segments,
            measure=TortuosityMeasure.Distance,
            length_measure=LengthMeasure.Splines,
            grid_field=ETDRS_FULL,
            aggregator=LengthWeightedAggregator(),
        ),
        Tortuosity(
            mode=TortuosityMode.Segments,
            max_segment_len=0.2,
            measure=TortuosityMeasure.Distance,
            length_measure=LengthMeasure.Splines,
            grid_field=ETDRS_FULL,
            aggregator=LengthWeightedAggregator(),
        ),
        Tortuosity(
            mode=TortuosityMode.Segments,
            measure=TortuosityMeasure.Curvature,
            length_measure=LengthMeasure.Splines,
            grid_field=ETDRS_FULL,
            aggregator=LengthWeightedAggregator(),
        ),

        # Disc region (length-weighted normalized)
        Tortuosity(
            mode=TortuosityMode.Segments,
            measure=TortuosityMeasure.Distance,
            length_measure=LengthMeasure.Splines,
            grid_field=DISC_FULL,
            aggregator=LengthWeightedAggregator(),
        ),
        Tortuosity(
            mode=TortuosityMode.Segments,
            max_segment_len=0.2,
            measure=TortuosityMeasure.Distance,
            length_measure=LengthMeasure.Splines,
            grid_field=DISC_FULL,
            aggregator=LengthWeightedAggregator(),
        ),
        Tortuosity(
            mode=TortuosityMode.Segments,
            measure=TortuosityMeasure.Curvature,
            length_measure=LengthMeasure.Splines,
            grid_field=DISC_FULL,
            aggregator=LengthWeightedAggregator(),
        ),

        # vascular densities (full, superior, inferior)
        VascularDensity(grid_field=ELLIPSE_FULL),
        VascularDensity(grid_field=DISC_FULL),
        VascularDensity(grid_field=ETDRS_FULL),
        VascularDensity(grid_field=HMF_SUP),
        VascularDensity(grid_field=HMF_INF),

        # disc–fovea distance
        DiscFoveaDistance(),

        ####  IMAGE QUALITY FEATURES ####

        # Sparsity features
        Sparsity(mode=SparsityMode.MEAN),
        Sparsity(mode=SparsityMode.MAX),
        Sparsity(
            mode=SparsityMode.MEAN, grid_field=ELLIPSE_FULL
        ),
        Sparsity(
            mode=SparsityMode.MAX, grid_field=ELLIPSE_FULL
        ),
        Sparsity(grid_field=DISC_FULL, mode=SparsityMode.MEAN),
        Sparsity(grid_field=DISC_FULL, mode=SparsityMode.MAX),
        Sparsity(grid_field=ETDRS_FULL, mode=SparsityMode.MEAN),
        Sparsity(grid_field=ETDRS_FULL, mode=SparsityMode.MAX),

        
    ],
)
