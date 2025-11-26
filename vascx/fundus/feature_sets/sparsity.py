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
    {
        # Sparsity features
        "sparsity_full_mean": Sparsity(mode=SparsityMode.MEAN),
        "sparsity_full_max": Sparsity(mode=SparsityMode.MAX),
        "sparsity_ellipse_mean": Sparsity(
            mode=SparsityMode.MEAN, grid_field=ELLIPSE_FULL
        ),
        "sparsity_ellipse_max": Sparsity(
            mode=SparsityMode.MAX, grid_field=ELLIPSE_FULL
        ),
        "sparsity_disc_mean": Sparsity(grid_field=DISC_FULL, mode=SparsityMode.MEAN),
        "sparsity_disc_max": Sparsity(grid_field=DISC_FULL, mode=SparsityMode.MAX),
        "sparsity_fovea_mean": Sparsity(grid_field=ETDRS_FULL, mode=SparsityMode.MEAN),
        "sparsity_fovea_max": Sparsity(grid_field=ETDRS_FULL, mode=SparsityMode.MAX),
        # Laplacian features (matching sparsity grid fields and naming)
        "lapl_full": VarianceOfLaplacian(),
        "lapl_ellipse": VarianceOfLaplacian(grid_field=ELLIPSE_FULL),
        "lapl_disc": VarianceOfLaplacian(grid_field=DISC_FULL),
        "lapl_fovea": VarianceOfLaplacian(grid_field=ETDRS_FULL),
        # Vascular density features (matching sparsity grid fields and naming)
        "vd_full": VascularDensity(),
        "vd_ellipse": VascularDensity(grid_field=ELLIPSE_FULL),
        "vd_disc": VascularDensity(grid_field=DISC_FULL),
        "vd_fovea": VascularDensity(grid_field=ETDRS_FULL),
    },
)
