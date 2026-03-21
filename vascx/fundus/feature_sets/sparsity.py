from rtnls_enface.grids.disc_centered import DiscCenteredRing
from rtnls_enface.grids.ellipse import EllipseField
from rtnls_enface.grids.etdrs import ETDRSRing as ETDRSRing
from rtnls_enface.grids.specifications import (
    DiscCenteredGridSpecification,
    EllipseGridSpecification,
    ETDRSGridSpecification,
    GridFieldSpecification,
)

from vascx.fundus.features.sparsity import Sparsity, SparsityMode
from vascx.fundus.features.variance_of_laplacian import VarianceOfLaplacian
from vascx.fundus.features.vascular_densities import VascularDensity
from vascx.shared.features import FeatureSet

DISC_FULL = GridFieldSpecification(
    DiscCenteredGridSpecification(multiplier=1.2), DiscCenteredRing.FullGrid
)
ELLIPSE_FULL = GridFieldSpecification(EllipseGridSpecification(), EllipseField.FullGrid)
ETDRS_FULL = GridFieldSpecification(
    ETDRSGridSpecification(multiplier=1.5), ETDRSRing.FullGrid
)


fs_sparsity = FeatureSet(
    "sparsity",
    [
        # Sparsity features
        Sparsity(mode=SparsityMode.MEAN),
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
        VarianceOfLaplacian(),
        VarianceOfLaplacian(grid_field=ELLIPSE_FULL),
        VarianceOfLaplacian(grid_field=DISC_FULL),
        VarianceOfLaplacian(grid_field=ETDRS_FULL),
        # VascularDensity() already uses the ellipse grid by default, so keep only one ellipse entry.
        VascularDensity(grid_field=ELLIPSE_FULL),
        VascularDensity(grid_field=DISC_FULL),
        VascularDensity(grid_field=ETDRS_FULL),
    ],
)
