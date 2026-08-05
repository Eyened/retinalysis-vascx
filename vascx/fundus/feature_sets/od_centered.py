# fmt: off
from rtnls_enface.grids.circle import CircleField
from rtnls_enface.grids.disc_centered import DiscCenteredQuadrant, DiscCenteredRing, DiscCenteredHemifield
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
from vascx.fundus.features.disc_features import DiscFoveaDistance, DiscFoveaDistanceMode
from vascx.fundus.features.sharpness import Sharpness
from vascx.fundus.features.sparsity import Sparsity, SparsityMode
from vascx.fundus.features.tortuosity import (
    LengthMeasure,
    Tortuosity,
    TortuosityMeasure,
    TortuosityMode,
)
from vascx.fundus.features.temporal_angles import TemporalAngle
from vascx.fundus.features.vascular_densities import VascularDensity
from vascx.shared.aggregators import LengthWeightedAggregator, mean, median
from vascx.shared.features import FeatureSet


# Feature set for OD-centered images where the fovea location is known
def make_set(name: str, description: str, multiplier: float=7 / 6, band_crop: bool=False, min_area_within_bounds: float=None) -> FeatureSet:
    DISC_GRID = DiscCenteredGridSpecification(multiplier=multiplier, band_crop_fraction=0.06 if band_crop else 0.0, name="crcl", min_area_within_bounds=min_area_within_bounds)
    DISC_FULL = GridFieldSpecification(
        grid_spec=DISC_GRID,
        field=DiscCenteredRing.FullGrid,
    )
    DISC_SUP = GridFieldSpecification(DISC_GRID, DiscCenteredHemifield.Superior)
    DISC_INF = GridFieldSpecification(DISC_GRID, DiscCenteredHemifield.Inferior)
    DISC_TEMP = GridFieldSpecification(DISC_GRID, DiscCenteredHemifield.Temporal)
    DISC_NASAL = GridFieldSpecification(DISC_GRID, DiscCenteredHemifield.Nasal)
    
    return FeatureSet(
        name,
        [
            TemporalAngle(),

            # bifurcation angles (full, superior, inferior)
            BifurcationAngles(grid_field=DISC_FULL, aggregator=mean),
            BifurcationAngles(grid_field=DISC_SUP, aggregator=mean),
            BifurcationAngles(grid_field=DISC_INF, aggregator=mean),
            BifurcationAngles(grid_field=DISC_TEMP, aggregator=mean),
            BifurcationAngles(grid_field=DISC_NASAL, aggregator=mean),

            # caliber (full, superior, inferior)
            Caliber(grid_field=DISC_FULL, aggregator=LengthWeightedAggregator()),
            Caliber(grid_field=DISC_SUP, aggregator=LengthWeightedAggregator()),
            Caliber(grid_field=DISC_INF, aggregator=LengthWeightedAggregator()),
            Caliber(grid_field=DISC_TEMP, aggregator=LengthWeightedAggregator()),
            Caliber(grid_field=DISC_NASAL, aggregator=LengthWeightedAggregator()),

            # CRE: temporal variants in sup/inf/full; nasal and full variants on full grid
            CRE(CREMode.Full, min_circles=2),
            CRE(CREMode.Nasal, min_circles=2),
            CRE(CREMode.Temporal, min_circles=2),

            # tortuosity (segments) — Distance and Curvature
            # whole image (length-weighted normalized)
            Tortuosity(
                mode=TortuosityMode.Segments,
                max_segment_len=0.15,
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
                max_segment_len=0.25,
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

            # tortuosity distance per region
            Tortuosity(
                mode=TortuosityMode.Segments,
                max_segment_len=0.2,
                measure=TortuosityMeasure.Distance,
                length_measure=LengthMeasure.Splines,
                grid_field=DISC_SUP,
                aggregator=LengthWeightedAggregator(),
            ),
            Tortuosity(
                mode=TortuosityMode.Segments,
                max_segment_len=0.2,
                measure=TortuosityMeasure.Distance,
                length_measure=LengthMeasure.Splines,
                grid_field=DISC_INF,
                aggregator=LengthWeightedAggregator(),
            ),
            Tortuosity(
                mode=TortuosityMode.Segments,
                max_segment_len=0.2,
                measure=TortuosityMeasure.Distance,
                length_measure=LengthMeasure.Splines,
                grid_field=DISC_TEMP,
                aggregator=LengthWeightedAggregator(),
            ),
            Tortuosity(
                mode=TortuosityMode.Segments,
                max_segment_len=0.2,
                measure=TortuosityMeasure.Distance,
                length_measure=LengthMeasure.Splines,
                grid_field=DISC_NASAL,
                aggregator=LengthWeightedAggregator(),
            ),

            # tortuosity curvature per region
            Tortuosity(
                mode=TortuosityMode.Segments,
                measure=TortuosityMeasure.Curvature,
                length_measure=LengthMeasure.Splines,
                grid_field=DISC_SUP,
                aggregator=LengthWeightedAggregator(),
            ),
            Tortuosity(
                mode=TortuosityMode.Segments,
                measure=TortuosityMeasure.Curvature,
                length_measure=LengthMeasure.Splines,
                grid_field=DISC_INF,
                aggregator=LengthWeightedAggregator(),
            ),
            Tortuosity(
                mode=TortuosityMode.Segments,
                measure=TortuosityMeasure.Curvature,
                length_measure=LengthMeasure.Splines,
                grid_field=DISC_TEMP,
                aggregator=LengthWeightedAggregator(),
            ),
            Tortuosity(
                mode=TortuosityMode.Segments,
                measure=TortuosityMeasure.Curvature,
                length_measure=LengthMeasure.Splines,
                grid_field=DISC_NASAL,
                aggregator=LengthWeightedAggregator(),
            ),

            # vascular densities (full, superior, inferior)
            VascularDensity(grid_field=DISC_FULL),
            VascularDensity(grid_field=DISC_SUP),
            VascularDensity(grid_field=DISC_INF),
            VascularDensity(grid_field=DISC_TEMP),
            VascularDensity(grid_field=DISC_NASAL),

            # disc–fovea distance
            DiscFoveaDistance(),
            DiscFoveaDistance(mode=DiscFoveaDistanceMode.Edge),
            ####  IMAGE QUALITY FEATURES ####

            # Sparsity features
            Sparsity(mode=SparsityMode.MEAN),
            Sparsity(
                mode=SparsityMode.MEAN, grid_field=DISC_FULL
            ),
            Sharpness(grid_field=DISC_FULL)
        
        ],
        description=description,
    )

fs_od_centered = make_set(
    name="od_centered",
    description="Biomarkers optimized for optic-disc–centered fundus images with a view of the fovea.",
    band_crop=False)

fs_od_centered_rs = make_set(
    name="od_centered_rs",
    description="Biomarkers for optic-disc–centered fundus images in the Rotterdam Study.",
    band_crop=True)

fs_od_centered_narrow_rs = make_set(
    name="od_centered_narrow_rs",
    multiplier=2/3,
    description="Biomarkers for very narrow optic-disc–centered fundus images in the Rotterdam Study.")
