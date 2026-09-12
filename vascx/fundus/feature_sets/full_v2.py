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
from vascx.fundus.features.bifurcation_counts import BifurcationCount
from vascx.fundus.features.caliber import Caliber
from vascx.fundus.features.cre import CRE, CREMode
from vascx.fundus.features.disc_features import DiscFoveaDistance
from vascx.fundus.features.sparsity import Sparsity, SparsityMode
from vascx.fundus.features.tortuosity import (
    LengthMeasure,
    Tortuosity,
    TortuosityMeasure,
    TortuosityMode,
)
from vascx.fundus.features.variance_of_laplacian import VarianceOfLaplacian
from vascx.fundus.features.vascular_densities import VascularDensity
from vascx.shared.aggregators import LengthWeightedAggregator, median
from vascx.shared.features import FeatureSet

HMF_SUP = GridFieldSpecification(HemifieldGridSpecification(description="the visible retina divided into superior and inferior regions by the optic-disc–fovea axis"), HemifieldField.Superior)
HMF_INF = GridFieldSpecification(HemifieldGridSpecification(description="the visible retina divided into superior and inferior regions by the optic-disc–fovea axis"), HemifieldField.Inferior)
DISC_FULL = GridFieldSpecification(DiscCenteredGridSpecification(description="an optic-disc-centered annulus extending from the disc margin to 0.6 times the optic-disc–fovea distance beyond it"), DiscCenteredRing.FullGrid)
ELLIPSE_FULL = GridFieldSpecification(EllipseGridSpecification(description="an ellipse centered midway between the optic disc and fovea, aligned with their axis and scaled to their separation"), EllipseField.FullGrid)
ETDRS_FULL = GridFieldSpecification(ETDRSGridSpecification(description="a fovea-centered ETDRS grid with ring radii of 0.5, 1.5, and 3.0 mm"), ETDRSRing.FullGrid)

fs_full_v2 = FeatureSet(
    "full_v2",
    [
        # bifurcation angles (full, superior, inferior)
        BifurcationAngles(plot=True, aggregator=median),
        BifurcationAngles(grid_field=HMF_SUP, aggregator=median),
        BifurcationAngles(grid_field=HMF_INF, aggregator=median),

        # bifurcation counts (full, superior, inferior)
        # Note: we have deprecated BifurcationCount due to low reproducibility scores.
        BifurcationCount(plot=True),
        BifurcationCount(grid_field=HMF_SUP),
        BifurcationCount(grid_field=HMF_INF),

        # caliber (full, superior, inferior)
        Caliber(plot=True, aggregator=median),
        Caliber(grid_field=HMF_SUP, aggregator=median),
        Caliber(grid_field=HMF_INF, aggregator=median),

        # coverage and variance of laplacian over disc-centered full grid
        Sparsity(plot=True, mode=SparsityMode.MEAN, grid_field=ELLIPSE_FULL),
        Sparsity(mode=SparsityMode.MAX, grid_field=ELLIPSE_FULL),
        Sparsity(grid_field=DISC_FULL, mode=SparsityMode.MEAN),
        Sparsity(grid_field=DISC_FULL, mode=SparsityMode.MAX),
        Sparsity(grid_field=ETDRS_FULL, mode=SparsityMode.MEAN),
        Sparsity(grid_field=ETDRS_FULL, mode=SparsityMode.MAX),


        VarianceOfLaplacian(plot=True),
        VarianceOfLaplacian(grid_field=DISC_FULL),
        VarianceOfLaplacian(grid_field=ETDRS_FULL),

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
        # whole image (length-weighted normalized)
        Tortuosity(
            mode=TortuosityMode.Segments,
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
        # ETDRS total (non-normalized median)
        Tortuosity(
            mode=TortuosityMode.Segments,
            measure=TortuosityMeasure.Distance,
            length_measure=LengthMeasure.Splines,
            grid_field=ETDRS_FULL,
            aggregator=median,
        ),
        Tortuosity(
            mode=TortuosityMode.Segments,
            measure=TortuosityMeasure.Curvature,
            length_measure=LengthMeasure.Splines,
            grid_field=ETDRS_FULL,
            aggregator=median,
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
            measure=TortuosityMeasure.Curvature,
            length_measure=LengthMeasure.Splines,
            grid_field=ETDRS_FULL,
            aggregator=LengthWeightedAggregator(),
        ),

        # vascular densities (full, superior, inferior)
        VascularDensity(plot=True),
        VascularDensity(grid_field=HMF_SUP),
        VascularDensity(grid_field=HMF_INF),

        # disc–fovea distance
        DiscFoveaDistance(plot=True),
    ],
    description=(
        "Previous comprehensive fundus biomarker set. Prefer full_v3 for new analyses."
    ),
)
