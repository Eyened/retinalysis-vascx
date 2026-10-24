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

from vascx.fundus.features.fractal_dimension import FractalDimension
from vascx.fundus.features.bifurcation import Bifurcation
from vascx.fundus.features.caliber import Caliber
from vascx.fundus.features.cre import CRE, CREMeasure, CREMode
from vascx.fundus.features.od_morphology import ODMorphology
from vascx.fundus.features.luminance import Luminance
from vascx.fundus.features.rps import BackgroundLab
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
from vascx.fundus.features.derived import AVR
from vascx.shared.aggregators import LengthWeightedAggregator
from vascx.shared.features import FeatureSet


# Feature set for OD-centered images where the fovea location is known
def make_set(name: str, description: str, grid_description: str, multiplier: float=7 / 6, band_crop: bool=False, min_area_within_bounds: float=None, include_hemifields: bool=False) -> FeatureSet:
    DISC_GRID = DiscCenteredGridSpecification(multiplier=multiplier, band_crop_fraction=0.06 if band_crop else 0.0, name="crcl", description=grid_description, min_area_within_bounds=min_area_within_bounds)
    DISC_FULL = GridFieldSpecification(
        grid_spec=DISC_GRID,
        field=DiscCenteredRing.FullGrid,
    )
    DISC_SUP = GridFieldSpecification(DISC_GRID, DiscCenteredHemifield.Superior)
    DISC_INF = GridFieldSpecification(DISC_GRID, DiscCenteredHemifield.Inferior)
    DISC_TEMP = GridFieldSpecification(DISC_GRID, DiscCenteredHemifield.Temporal)
    DISC_NASAL = GridFieldSpecification(DISC_GRID, DiscCenteredHemifield.Nasal)
    
    features = [
        TemporalAngle(plot=True),

        # bifurcation features (full, superior, inferior)
        Bifurcation(plot=True, grid_field=DISC_FULL),
        *([
            Bifurcation(grid_field=DISC_SUP),
            Bifurcation(grid_field=DISC_INF),
            Bifurcation(grid_field=DISC_TEMP),
            Bifurcation(grid_field=DISC_NASAL)
        ] if include_hemifields else []),

        # caliber (full, superior, inferior)
        Caliber(plot=True, grid_field=DISC_FULL, aggregator=LengthWeightedAggregator()),
        *([
            Caliber(grid_field=DISC_SUP, aggregator=LengthWeightedAggregator()),
            Caliber(grid_field=DISC_INF, aggregator=LengthWeightedAggregator()),
            Caliber(grid_field=DISC_TEMP, aggregator=LengthWeightedAggregator()),
            Caliber(grid_field=DISC_NASAL, aggregator=LengthWeightedAggregator())
        ] if include_hemifields else []),

        # CRE: full, nasal, and temporal regions
        CRE(CREMode.Full, plot=True, min_circles=2),
        *([
            CRE(CREMode.Nasal, min_circles=2),
            CRE(CREMode.Temporal, min_circles=2)
        ] if include_hemifields else []),

        # ARIC-style trunks; regional variants extend the full-field protocol.
        CRE(CREMode.Full, measure=CREMeasure.ARIC, plot=True),
        *([
            CRE(CREMode.Nasal, measure=CREMeasure.ARIC),
            CRE(CREMode.Temporal, measure=CREMeasure.ARIC),
            CRE(CREMode.Full, measure=CREMeasure.ARIC, hemifield=HemifieldField.Superior),
            CRE(CREMode.Full, measure=CREMeasure.ARIC, hemifield=HemifieldField.Inferior)
        ] if include_hemifields else []),

        # tortuosity (segments) — Distance and Curvature
        # whole image (length-weighted normalized)
        Tortuosity(plot=True,
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

        # tortuosity distance per region
        *([
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
        ] if include_hemifields else []),

        # vascular densities (full, superior, inferior)
        VascularDensity(plot=True, grid_field=DISC_FULL),
        *([
            VascularDensity(grid_field=DISC_SUP),
            VascularDensity(grid_field=DISC_INF),
            VascularDensity(grid_field=DISC_TEMP),
            VascularDensity(grid_field=DISC_NASAL)
        ] if include_hemifields else []),

        FractalDimension(q=0, grid_field=DISC_FULL, plot=True),

        # optic disc features
        ODMorphology(plot=True),
        ####  IMAGE QUALITY FEATURES ####

        # Sparsity features
        Sparsity(plot=True, mode=SparsityMode.MEAN),
        Sparsity(
            mode=SparsityMode.MEAN, grid_field=DISC_FULL
        ),
        Sharpness(plot=True, grid_field=DISC_FULL),
        Luminance(plot=True),
        BackgroundLab(plot=True),
    ]
    features.extend(AVR(source) for source in list(features)
                    if isinstance(source, CRE))
    return FeatureSet(name, features, description=description)


fs_od_centered = make_set(
    name="od_centered",
    description="Biomarkers optimized for optic-disc–centered fundus images with a view of the fovea.",
    grid_description="an optic-disc-centered annulus extending 0.7 times the optic-disc–fovea distance beyond the disc margin",
    band_crop=False)

fs_od_centered_hemifields = make_set(
    name="od_centered_hemifields",
    description="od_centered feature set expanded with hemifield features",
    grid_description="an optic-disc-centered annulus extending 0.7 times the optic-disc–fovea distance beyond the disc margin",
    band_crop=False,
    include_hemifields=True)

fs_od_centered_rs = make_set(
    name="od_centered_rs",
    description="Biomarkers for optic-disc–centered fundus images in the Rotterdam Study.",
    grid_description="a cropped optic-disc-centered annulus extending 0.7 times the optic-disc–fovea distance beyond the disc margin for Rotterdam Study images",
    band_crop=True)

fs_od_centered_narrow_rs = make_set(
    name="od_centered_narrow_rs",
    multiplier=2/3,
    description="Biomarkers for very narrow optic-disc–centered fundus images in the Rotterdam Study.",
    grid_description="a compact optic-disc-centered annulus extending 0.4 times the optic-disc–fovea distance beyond the disc margin")
