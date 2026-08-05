# fmt: off
from rtnls_enface.grids.circle import CircleField
from rtnls_enface.grids.etdrs import ETDRSRing
from rtnls_enface.grids.hemifields import HemifieldField
from rtnls_enface.grids.specifications import (
    CircleGridSpecification,
    ETDRSGridSpecification,
    GridFieldSpecification,
)

from vascx.fundus.features.bifurcation_angles import BifurcationAngles
from vascx.fundus.features.caliber import Caliber
from vascx.fundus.features.cre import CRE, CREMode
from vascx.fundus.features.disc_features import DiscFoveaDistance, DiscFoveaDistanceMode
from vascx.fundus.features.sharpness import Sharpness
from vascx.fundus.features.sparsity import Sparsity, SparsityMode
from vascx.fundus.features.temporal_angles import TemporalAngle
from vascx.fundus.features.tortuosity import (
    LengthMeasure,
    Tortuosity,
    TortuosityMeasure,
    TortuosityMode,
)
from vascx.fundus.features.vascular_densities import VascularDensity
from vascx.shared.aggregators import LengthWeightedAggregator, mean
from vascx.shared.features import FeatureSet



def make_set(name: str, description: str, center: float=1.0, radius_multiplier: float=1.0, band_crop: bool=False) -> FeatureSet:

    ETDRS_FULL = GridFieldSpecification(ETDRSGridSpecification(), ETDRSRing.FullGrid)

    CIRCLE_CROPPED_GRID = CircleGridSpecification(
        band_crop_fraction=0.12 if band_crop else 0.0, center=center, radius_multiplier=radius_multiplier, name="crcl"
    )
    CIRCLE_CROPPED_FIELD = GridFieldSpecification(CIRCLE_CROPPED_GRID, CircleField.FullGrid)
    CIRCLE_CROPPED_SUP = GridFieldSpecification(CIRCLE_CROPPED_GRID, CircleField.Superior)
    CIRCLE_CROPPED_INF = GridFieldSpecification(CIRCLE_CROPPED_GRID, CircleField.Inferior)

    return FeatureSet(
        name,
        [
            TemporalAngle(),

            # bifurcation angles (full, superior, inferior)
            BifurcationAngles(aggregator=mean, grid_field=CIRCLE_CROPPED_FIELD),
            BifurcationAngles(grid_field=CIRCLE_CROPPED_SUP, aggregator=mean),
            BifurcationAngles(grid_field=CIRCLE_CROPPED_INF, aggregator=mean),

            # caliber (length-weighted)
            Caliber(grid_field=CIRCLE_CROPPED_FIELD, aggregator=LengthWeightedAggregator()),
            Caliber(grid_field=CIRCLE_CROPPED_SUP, aggregator=LengthWeightedAggregator()),
            Caliber(grid_field=CIRCLE_CROPPED_INF, aggregator=LengthWeightedAggregator()),
            Caliber(grid_field=ETDRS_FULL, aggregator=LengthWeightedAggregator()),

            # CRE: temporal variants in sup/inf/full; nasal and full variants on full grid
            CRE(CREMode.Temporal),
            CRE(CREMode.Temporal, max_vessels=3),
            CRE(CREMode.Temporal, max_vessels=6),
            CRE(CREMode.Temporal, hemifield=HemifieldField.Superior),
            CRE(CREMode.Temporal, hemifield=HemifieldField.Inferior),

            # tortuosity (segments) — Distance and Curvature
            # whole image (length-weighted normalized)
            Tortuosity(
                mode=TortuosityMode.Segments,
                max_segment_len=0.2,
                measure=TortuosityMeasure.Distance,
                length_measure=LengthMeasure.Splines,
                grid_field=CIRCLE_CROPPED_FIELD,
                aggregator=LengthWeightedAggregator(),
            ),
            Tortuosity(
                mode=TortuosityMode.Segments,
                measure=TortuosityMeasure.Curvature,
                length_measure=LengthMeasure.Splines,
                grid_field=CIRCLE_CROPPED_FIELD,
                aggregator=LengthWeightedAggregator(),
            ),
            # ETDRS total (length-weighted normalized)
            Tortuosity(
                mode=TortuosityMode.Segments,
                max_segment_len=0.15,
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
                max_segment_len=0.25,
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
            VascularDensity(grid_field=CIRCLE_CROPPED_FIELD),
            VascularDensity(grid_field=CIRCLE_CROPPED_SUP),
            VascularDensity(grid_field=CIRCLE_CROPPED_INF),

            # disc–fovea distance
            DiscFoveaDistance(),
            DiscFoveaDistance(mode=DiscFoveaDistanceMode.Edge),
            ####  IMAGE QUALITY FEATURES ####

            # Sparsity features
            Sparsity(mode=SparsityMode.MEAN),
            Sparsity(
                mode=SparsityMode.MEAN, grid_field=CIRCLE_CROPPED_FIELD
            ),
            Sparsity(
                mode=SparsityMode.MEAN, grid_field=ETDRS_FULL
            ),
            Sharpness(grid_field=CIRCLE_CROPPED_FIELD)
        ],
        description=description,
    )


fs_macula_centered = make_set(
    name="macula_centered",
    description="Biomarkers optimized for macula-centered fundus images.")
fs_macula_centered_rs = make_set(
    name="macula_centered_rs", 
    description="Biomarkers for macula-centered fundus images using a cropped circular field optimized for devices in the Rotterdam Study (narrow FoV and band crop).",
    center=0.9,
    radius_multiplier=0.9,
    band_crop=True)