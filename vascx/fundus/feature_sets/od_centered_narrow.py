# fmt: off
from rtnls_enface.grids.od_only import ODOnlyField
from rtnls_enface.grids.specifications import (
    GridFieldSpecification,
    ODOnlyGridSpecification,
)

from vascx.fundus.features.bifurcation_angles import BifurcationAngles
from vascx.fundus.features.caliber import Caliber
from vascx.fundus.features.cre import CRE, CREMode
from vascx.fundus.features.luminance import Luminance
from vascx.fundus.features.sharpness import Sharpness
from vascx.fundus.features.sparsity import Sparsity, SparsityMode
from vascx.fundus.features.tortuosity import (
    LengthMeasure,
    Tortuosity,
    TortuosityMeasure,
    TortuosityMode,
)
from vascx.fundus.features.vascular_densities import VascularDensity
from vascx.shared.aggregators import LengthWeightedAggregator, mean
from vascx.shared.features import FeatureSet


# Feature set for OD-centered images where the fovea location is not known.
# Uses an axis-aligned OD-only circle (no OD–fovea geometry).
def make_set(
    name: str,
    description: str,
    radius_px: float = 200.0,
    min_area_within_bounds: float = None,
) -> FeatureSet:
    OD_GRID = ODOnlyGridSpecification(
        radius_px=radius_px,
        name="od",
        description="a circular grid with a 200-pixel radius centered on the optic disc, for images without visible fovea",
        min_area_within_bounds=min_area_within_bounds,
    )
    OD_FULL = GridFieldSpecification(OD_GRID, ODOnlyField.FullGrid)
    OD_SUP = GridFieldSpecification(OD_GRID, ODOnlyField.Superior)
    OD_INF = GridFieldSpecification(OD_GRID, ODOnlyField.Inferior)
    OD_LEFT = GridFieldSpecification(OD_GRID, ODOnlyField.Left)
    OD_RIGHT = GridFieldSpecification(OD_GRID, ODOnlyField.Right)

    return FeatureSet(
        name,
        [

            # caliber (full, superior, inferior, left, right)
            Caliber(plot=True, grid_field=OD_FULL, aggregator=LengthWeightedAggregator()),
            Caliber(grid_field=OD_SUP, aggregator=LengthWeightedAggregator()),
            Caliber(grid_field=OD_INF, aggregator=LengthWeightedAggregator()),
            Caliber(grid_field=OD_LEFT, aggregator=LengthWeightedAggregator()),
            Caliber(grid_field=OD_RIGHT, aggregator=LengthWeightedAggregator()),

            # CRE: full mode only (temporal/nasal require fovea)
            CRE(CREMode.Full, plot=True, inner_circle=0.8, outer_circle=1.275, min_circles=2),

            # tortuosity (segments) — Curvature
            Tortuosity(plot=True,
                mode=TortuosityMode.Segments,
                measure=TortuosityMeasure.Curvature,
                length_measure=LengthMeasure.Splines,
                grid_field=OD_FULL,
                aggregator=LengthWeightedAggregator(),
            ),
            # tortuosity curvature per region
            Tortuosity(
                mode=TortuosityMode.Segments,
                measure=TortuosityMeasure.Curvature,
                length_measure=LengthMeasure.Splines,
                grid_field=OD_SUP,
                aggregator=LengthWeightedAggregator(),
            ),
            Tortuosity(
                mode=TortuosityMode.Segments,
                measure=TortuosityMeasure.Curvature,
                length_measure=LengthMeasure.Splines,
                grid_field=OD_INF,
                aggregator=LengthWeightedAggregator(),
            ),
            Tortuosity(
                mode=TortuosityMode.Segments,
                measure=TortuosityMeasure.Curvature,
                length_measure=LengthMeasure.Splines,
                grid_field=OD_LEFT,
                aggregator=LengthWeightedAggregator(),
            ),
            Tortuosity(
                mode=TortuosityMode.Segments,
                measure=TortuosityMeasure.Curvature,
                length_measure=LengthMeasure.Splines,
                grid_field=OD_RIGHT,
                aggregator=LengthWeightedAggregator(),
            ),

            # vascular densities (full, superior, inferior, left, right)
            VascularDensity(plot=True, grid_field=OD_FULL),
            VascularDensity(grid_field=OD_SUP),
            VascularDensity(grid_field=OD_INF),
            VascularDensity(grid_field=OD_LEFT),
            VascularDensity(grid_field=OD_RIGHT),

            ####  IMAGE QUALITY FEATURES ####
            # normalize=False: disc–fovea distance is unavailable
            Sparsity(plot=True, mode=SparsityMode.MEAN, normalize=False),
            Sparsity(mode=SparsityMode.MEAN, grid_field=OD_FULL, normalize=False),
            Sharpness(plot=True, grid_field=OD_FULL),
            Luminance(plot=True),
        ],
        description=description,
    )


fs_od_centered_narrow = make_set(
    name="od_centered_narrow",
    description=(
        "Biomarkers for optic-disc–centered images with narrow field of view "
        "for which the fovea is out of bounds."
    ),
    min_area_within_bounds=0.95
)
