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
from vascx.fundus.features.sparsity import Sparsity, SparsityMode
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


# Feature set for OD-centered images where the fovea location is not known
# ie. none of the following features use localizers relative to the fovea
fs_od_centered = FeatureSet(
    "od_centered",
    [
        # bifurcation angles (full, superior, inferior)
        BifurcationAngles(aggregator=mean),

        # caliber (full, superior, inferior)
        Caliber(aggregator=median),
        Caliber(aggregator=LengthWeightedAggregator()),

        # CRE: temporal variants in sup/inf/full; nasal and full variants on full grid
        CRE(CREMode.Full, inner_circle=0.85, outer_circle=1.275, min_circles=3),

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

        # vascular densities (full, superior, inferior)
        VascularDensity(),

        # Sparsity features
        Sparsity(mode=SparsityMode.MEAN, normalize=False),
    ],
)
