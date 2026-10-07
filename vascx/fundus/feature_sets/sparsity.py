from rtnls_enface.grids.disc_centered import DiscCenteredRing
from rtnls_enface.grids.ellipse import EllipseField
from rtnls_enface.grids.etdrs import ETDRSRing as ETDRSRing
from rtnls_enface.grids.specifications import (
    DiscCenteredGridSpecification,
    EllipseGridSpecification,
    ETDRSGridSpecification,
    GridFieldSpecification,
)

from vascx.fundus.features.fractal_dimension import FractalDimension
from vascx.fundus.features.sparsity import Sparsity, SparsityMode
from vascx.fundus.features.variance_of_laplacian import VarianceOfLaplacian
from vascx.fundus.features.vascular_densities import VascularDensity
from vascx.shared.features import FeatureSet

DISC_FULL = GridFieldSpecification(
    DiscCenteredGridSpecification(multiplier=1.2, description="an optic-disc-centered annulus extending 0.72 times the optic-disc–fovea distance beyond the disc margin"), DiscCenteredRing.FullGrid
)
ELLIPSE_FULL = GridFieldSpecification(EllipseGridSpecification(description="an ellipse centered midway between the optic disc and fovea, aligned with their axis and scaled to their separation"), EllipseField.FullGrid)
ETDRS_FULL = GridFieldSpecification(
    ETDRSGridSpecification(multiplier=1.5, description="a fovea-centered ETDRS grid with ring radii of 0.75, 2.25, and 4.5 mm"), ETDRSRing.FullGrid
)


fs_sparsity = FeatureSet(
    "sparsity",
    [
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
        # Laplacian features (matching sparsity grid fields and naming)
        VarianceOfLaplacian(plot=True),
        VarianceOfLaplacian(grid_field=ELLIPSE_FULL),
        VarianceOfLaplacian(grid_field=DISC_FULL),
        VarianceOfLaplacian(grid_field=ETDRS_FULL),
        VascularDensity(plot=True),
        VascularDensity(grid_field=ELLIPSE_FULL),
        VascularDensity(grid_field=DISC_FULL),
        VascularDensity(grid_field=ETDRS_FULL),
        # Three scalar outputs share one calculation and one report panel per region.
        *[
            FractalDimension(q=q, grid_field=region, plot=(q == 0))
            for region in (None, ELLIPSE_FULL, DISC_FULL, ETDRS_FULL)
            for q in (0, 1, 2)
        ],
    ],
    description=(
        "Experimental set of vessel sparsity biomarkers with matching variance-of-Laplacian "
        "and vascular density measures, plus generalized fractal dimensions D0/D1/D2, "
        "on the whole retina and ellipse, disc-centered, and ETDRS grids."
    ),
)
