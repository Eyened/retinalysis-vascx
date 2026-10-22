"""Matched full-circle and superior/inferior measurements for fundus images."""

from rtnls_enface.grids.circle import CircleField
from rtnls_enface.grids.specifications import (
    CircleGridSpecification,
    GridFieldSpecification,
)

from vascx.fundus.features.bifurcation import Bifurcation
from vascx.fundus.features.caliber import Caliber
from vascx.fundus.features.luminance import Luminance
from vascx.fundus.features.rps import BackgroundLab
from vascx.fundus.features.sharpness import Sharpness
from vascx.fundus.features.tortuosity import (
    LengthMeasure,
    Tortuosity,
    TortuosityMeasure,
    TortuosityMode,
)
from vascx.fundus.features.vascular_densities import VascularDensity
from vascx.shared.aggregators import LengthWeightedAggregator, median, std
from vascx.shared.features import FeatureSet


def make_set(
    name: str,
    description: str,
    grid_description: str,
    center: float = 1.0,
    radius_multiplier: float = 1.0,
    band_crop: bool = False,
) -> FeatureSet:
    """Use the macula-centered circle geometry for all three measurement regions."""
    grid = CircleGridSpecification(
        center=center,
        radius_multiplier=radius_multiplier,
        band_crop_fraction=0.12 if band_crop else 0.0,
        name="crcl",
        description=grid_description,
    )
    features = []
    for field in (CircleField.FullGrid, CircleField.Superior, CircleField.Inferior):
        region = GridFieldSpecification(grid, field)
        features.extend([
            VascularDensity(grid_field=region),
            Caliber(grid_field=region, aggregator=median),
            Caliber(grid_field=region, aggregator=std),
            Bifurcation(grid_field=region),
            Tortuosity(
                plot=True,
                mode=TortuosityMode.Segments,
                max_segment_len=0.2,
                measure=TortuosityMeasure.Distance,
                length_measure=LengthMeasure.Splines,
                grid_field=region,
                aggregator=LengthWeightedAggregator(),
            ),
            Luminance(grid_field=region),
            BackgroundLab(grid_field=region),
            Sharpness(grid_field=region),
        ])
    return FeatureSet(name, features, description=description)


fs_hemifields = make_set(
    name="hemifields",
    description=(
        "Vascular and image biomarkers in the full macula-centered circle and its "
        "superior and inferior hemifields, divided along the optic-disc–fovea axis."
    ),
    grid_description=(
        "a circular grid centered on the fovea with radius equal to the "
        "optic-disc–fovea distance plus the optic-disc radius"
    ),
)

fs_hemifields_rs = make_set(
    name="hemifields_rs",
    description=(
        "Vascular and image biomarkers in the cropped macula-centered circle "
        "optimized for Rotterdam Study images and its superior and inferior "
        "hemifields, divided along the optic-disc–fovea axis."
    ),
    grid_description=(
        "a cropped circular grid centered nine-tenths of the way from the optic "
        "disc to the fovea, with radius equal to nine-tenths of their separation "
        "plus the optic-disc radius"
    ),
    center=0.9,
    radius_multiplier=0.9,
    band_crop=True,
)
