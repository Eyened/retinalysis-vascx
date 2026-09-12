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

HMF_SUP = GridFieldSpecification(HemifieldGridSpecification(description="the visible retina divided into superior and inferior regions by the optic-disc–fovea axis"), HemifieldField.Superior)
HMF_INF = GridFieldSpecification(HemifieldGridSpecification(description="the visible retina divided into superior and inferior regions by the optic-disc–fovea axis"), HemifieldField.Inferior)
DISC_FULL = GridFieldSpecification(DiscCenteredGridSpecification(description="an optic-disc-centered annulus extending from the disc margin to 0.6 times the optic-disc–fovea distance beyond it"), DiscCenteredRing.FullGrid)
ELLIPSE_FULL = GridFieldSpecification(EllipseGridSpecification(description="an ellipse centered midway between the optic disc and fovea, aligned with their axis and scaled to their separation"), EllipseField.FullGrid)
ETDRS_FULL = GridFieldSpecification(ETDRSGridSpecification(description="a fovea-centered ETDRS grid with ring radii of 0.5, 1.5, and 3.0 mm"), ETDRSRing.FullGrid)

fs_full_v3 = FeatureSet(
    "full_v3",
    [
        TemporalAngle(plot=True),

        # bifurcation angles (full, superior, inferior)
        BifurcationAngles(plot=True, aggregator=mean),
        BifurcationAngles(grid_field=HMF_SUP, aggregator=mean),
        BifurcationAngles(grid_field=HMF_INF, aggregator=mean),

        # caliber (full, superior, inferior)
        Caliber(plot=True, aggregator=median),
        Caliber(grid_field=HMF_SUP, aggregator=median),
        Caliber(grid_field=HMF_INF, aggregator=median),

        # caliber (length-weighted)
        Caliber(aggregator=LengthWeightedAggregator()),
        Caliber(grid_field=HMF_SUP, aggregator=LengthWeightedAggregator()),
        Caliber(grid_field=HMF_INF, aggregator=LengthWeightedAggregator()),
        Caliber(grid_field=DISC_FULL, aggregator=LengthWeightedAggregator()),

        # CRE: temporal variants in sup/inf/full; nasal and full variants on full grid
        CRE(CREMode.Temporal, plot=True),
        CRE(CREMode.Temporal, hemifield=HemifieldField.Superior),
        CRE(CREMode.Temporal, hemifield=HemifieldField.Inferior),
        CRE(CREMode.Nasal),
        CRE(CREMode.Full),

        # tortuosity (segments) — Distance and Curvature
        # whole image (non-normalized median)
        Tortuosity(plot=True,
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
        VascularDensity(plot=True, grid_field=ELLIPSE_FULL),
        VascularDensity(grid_field=DISC_FULL),
        VascularDensity(grid_field=ETDRS_FULL),
        VascularDensity(grid_field=HMF_SUP),
        VascularDensity(grid_field=HMF_INF),

        # disc–fovea distance
        DiscFoveaDistance(plot=True),

        ####  IMAGE QUALITY FEATURES ####

        # Sparsity features
        Sparsity(plot=True, mode=SparsityMode.MEAN),
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
    description=(
        "Recommended comprehensive fundus biomarker set covering caliber, CRE, "
        "tortuosity, sparsity, vascular density, and related measures on hemifield, "
        "disc-centered, ETDRS, and ellipse grids."
    ),
)
