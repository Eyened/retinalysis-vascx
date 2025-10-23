# fmt: off
from rtnls_enface.grids.etdrs import MiscField as ETDRS_MiscField
from rtnls_enface.grids.hemifields import HemifieldField
from rtnls_enface.grids.disc_centered import MiscField as DC_MiscField

from vascx.fundus.features.bifurcation_angles import BifurcationAngles
from vascx.fundus.features.bifurcation_counts import BifurcationCount
from vascx.fundus.features.caliber import Caliber
from vascx.fundus.features.coverage import Coverage
from vascx.fundus.features.cre import CRE, CREMode
from vascx.fundus.features.disc_features import DiscFoveaDistance
from vascx.fundus.features.tortuosity import (
    Tortuosity,
    TortuosityMode,
    TortuosityMeasure,
    LengthMeasure,
)
from vascx.fundus.features.variance_of_laplacian import VarianceOfLaplacian
from vascx.fundus.features.vascular_densities import VascularDensity
from vascx.shared.aggregators import mean_median, median, median_std, sum
from vascx.shared.features import FeatureSet


full_v2 = FeatureSet(
    "full_v2",
    {
        # bifurcation angles (full, superior, inferior)
        "bif_angles": BifurcationAngles(aggregator=median),
        "bif_angles_sup": BifurcationAngles(grid_field=HemifieldField.Superior, aggregator=median),
        "bif_angles_inf": BifurcationAngles(grid_field=HemifieldField.Inferior, aggregator=median),

        # bifurcation counts (full, superior, inferior)
        "bif": BifurcationCount(),
        "bif_sup": BifurcationCount(grid_field=HemifieldField.Superior),
        "bif_inf": BifurcationCount(grid_field=HemifieldField.Inferior),

        # caliber (full, superior, inferior)
        "diam": Caliber(aggregator=median_std),
        "diam_sup": Caliber(grid_field=HemifieldField.Superior, aggregator=median_std),
        "diam_inf": Caliber(grid_field=HemifieldField.Inferior, aggregator=median_std),

        # coverage and variance of laplacian over disc-centered full grid
        "coverage_dc": Coverage(grid_field=DC_MiscField.Grid),
        "lapl_dc": VarianceOfLaplacian(grid_field=DC_MiscField.Grid),

        # CRE: temporal variants in sup/inf/full; nasal and full variants on full grid
        "cre_temporal": CRE(CREMode.Temporal),
        "cre_temporal_sup": CRE(CREMode.Temporal, hemifield=HemifieldField.Superior),
        "cre_temporal_inf": CRE(CREMode.Temporal, hemifield=HemifieldField.Inferior),
        "cre_nasal": CRE(CREMode.Nasal),
        "cre_full": CRE(CREMode.Full),

        # tortuosity (segments) — Distance and Curvature
        # whole image (non-normalized median)
        "tort_dist": Tortuosity(
            mode=TortuosityMode.Segments,
            measure=TortuosityMeasure.Distance,
            length_measure=LengthMeasure.Splines,
            aggregator=median,
        ),
        "tort_curv": Tortuosity(
            mode=TortuosityMode.Segments,
            measure=TortuosityMeasure.Curvature,
            length_measure=LengthMeasure.Splines,
            aggregator=median,
        ),
        # whole image (length-weighted normalized)
        "tort_dist_norm": Tortuosity(
            mode=TortuosityMode.Segments,
            measure=TortuosityMeasure.Distance,
            length_measure=LengthMeasure.Splines,
            norm_measure=LengthMeasure.Splines,
            aggregator=sum,
        ),
        "tort_curv_norm": Tortuosity(
            mode=TortuosityMode.Segments,
            measure=TortuosityMeasure.Curvature,
            length_measure=LengthMeasure.Splines,
            norm_measure=LengthMeasure.Splines,
            aggregator=sum,
        ),
        # ETDRS total (non-normalized median)
        "tort_dist_etdrs": Tortuosity(
            mode=TortuosityMode.Segments,
            measure=TortuosityMeasure.Distance,
            length_measure=LengthMeasure.Splines,
            grid_field=ETDRS_MiscField.Total,
            aggregator=median,
        ),
        "tort_curv_etdrs": Tortuosity(
            mode=TortuosityMode.Segments,
            measure=TortuosityMeasure.Curvature,
            length_measure=LengthMeasure.Splines,
            grid_field=ETDRS_MiscField.Total,
            aggregator=median,
        ),
        # ETDRS total (length-weighted normalized)
        "tort_dist_etdrs_norm": Tortuosity(
            mode=TortuosityMode.Segments,
            measure=TortuosityMeasure.Distance,
            length_measure=LengthMeasure.Splines,
            grid_field=ETDRS_MiscField.Total,
            norm_measure=LengthMeasure.Splines,
            aggregator=sum,
        ),
        "tort_curv_etdrs_norm": Tortuosity(
            mode=TortuosityMode.Segments,
            measure=TortuosityMeasure.Curvature,
            length_measure=LengthMeasure.Splines,
            grid_field=ETDRS_MiscField.Total,
            norm_measure=LengthMeasure.Splines,
            aggregator=sum,
        ),

        # vascular densities (full, superior, inferior)
        "vd": VascularDensity(),
        "vd_sup": VascularDensity(grid_field=HemifieldField.Superior),
        "vd_inf": VascularDensity(grid_field=HemifieldField.Inferior),

        # disc–fovea distance
        "odfd": DiscFoveaDistance(),
    },
)


