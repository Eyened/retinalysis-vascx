# fmt: off
from rtnls_enface.grids.circle import CircleField
from rtnls_enface.grids.etdrs import ETDRSRing
from rtnls_enface.grids.hemifields import HemifieldField
from rtnls_enface.grids.specifications import (
    CircleGridSpecification,
    ETDRSGridSpecification,
    GridFieldSpecification,
)

from vascx.fundus.features.fractal_dimension import FractalDimension
from vascx.fundus.features.bifurcation import Bifurcation
from vascx.fundus.features.caliber import Caliber
from vascx.fundus.features.cre import CRE, CREMeasure, CREMode
from vascx.fundus.features.od_morphology import ODMorphology
from vascx.fundus.features.luminance import Luminance
from vascx.fundus.features.rps import BackgroundLab
from vascx.fundus.features.sharpness import Sharpness
from vascx.fundus.features.sparsity import Sparsity, SparsityMode
from vascx.fundus.features.temporal_angles import TemporalAngle
from vascx.fundus.features.tortuosity import (
    LengthMeasure,
    Tortuosity,
    TortuosityMeasure,
    TortuosityMode,
)
from vascx.fundus.features.derived import AVR
from vascx.fundus.features.vascular_densities import VascularDensity
from vascx.shared.aggregators import LengthWeightedAggregator
from vascx.shared.features import FeatureSet



def make_set(name: str, description: str, grid_description: str, center: float=1.0, radius_multiplier: float=1.0, band_crop: bool=False) -> FeatureSet:

    ETDRS_FULL = GridFieldSpecification(ETDRSGridSpecification(description="a fovea-centered ETDRS grid with ring radii of 0.5, 1.5, and 3.0 mm"), ETDRSRing.FullGrid)

    CIRCLE_CROPPED_GRID = CircleGridSpecification(
        band_crop_fraction=0.12 if band_crop else 0.0, center=center, radius_multiplier=radius_multiplier, name="crcl", description=grid_description
    )
    CIRCLE_CROPPED_FIELD = GridFieldSpecification(CIRCLE_CROPPED_GRID, CircleField.FullGrid)

    features = [
        TemporalAngle(plot=True),

        # bifurcation features (full, superior, inferior)
        Bifurcation(plot=True, grid_field=CIRCLE_CROPPED_FIELD),

        # caliber (length-weighted)
        Caliber(plot=True, grid_field=CIRCLE_CROPPED_FIELD, aggregator=LengthWeightedAggregator()),
        Caliber(grid_field=ETDRS_FULL, aggregator=LengthWeightedAggregator()),


        # CRE: temporal count variants and standalone superior/inferior regions
        CRE(CREMode.Temporal, plot=True),
        CRE(CREMode.Temporal, max_vessels=3),
        CRE(CREMode.Temporal, max_vessels=6),

        # ARIC-style trunks; regional variants extend the full-field protocol.
        CRE(CREMode.Temporal, measure=CREMeasure.ARIC),

        # tortuosity (segments) — Distance and Curvature
        # whole image (length-weighted normalized)
        Tortuosity(plot=True,
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

        # vascular densities (full, superior, inferior)
        VascularDensity(plot=True, grid_field=CIRCLE_CROPPED_FIELD, min_area_within_bounds=0.95),
        FractalDimension(q=0, grid_field=CIRCLE_CROPPED_FIELD, plot=True),

        # optic disc features
        ODMorphology(plot=True),
        ####  IMAGE QUALITY FEATURES ####

        # Sparsity features
        Sparsity(plot=True, mode=SparsityMode.MEAN),
        Sparsity(
            mode=SparsityMode.MEAN, grid_field=CIRCLE_CROPPED_FIELD
        ),
        Sparsity(
            mode=SparsityMode.MEAN, grid_field=ETDRS_FULL
        ),
        Sharpness(plot=True, grid_field=CIRCLE_CROPPED_FIELD),
        Sharpness(grid_field=ETDRS_FULL),
        Luminance(plot=True),
        BackgroundLab(plot=True),
    ]
    sources = [feature for feature in features if isinstance(feature, CRE)
               and feature.max_vessels == feature.default_max_vessels()]
    features.extend(AVR(source) for source in sources)
    return FeatureSet(name, features, description=description)


fs_macula_centered = make_set(
    name="macula_centered",
    description="Biomarkers optimized for macula-centered fundus images.",
    grid_description="a circular grid centered on the fovea with radius equal to the optic-disc–fovea distance plus the optic-disc radius")

fs_macula_centered_rs = make_set(
    name="macula_centered_rs", 
    description="Biomarkers for macula-centered fundus images using a cropped circular field optimized for devices in the Rotterdam Study (narrow FoV and band crop).",
    grid_description="a cropped circular grid centered nine-tenths of the way from the optic disc to the fovea, with radius equal to nine-tenths of their separation plus the optic-disc radius",
    center=0.9,
    radius_multiplier=0.9,
    band_crop=True)