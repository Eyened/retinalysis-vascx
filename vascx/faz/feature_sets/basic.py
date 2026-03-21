from rtnls_enface.grids.etdrs import ETDRSQuadrant, ETDRSRing
from rtnls_enface.grids.specifications import (
    ETDRSGridSpecification,
    GridFieldSpecification,
)
from vascx.faz.features.bifurcations import BifurcationCount
from vascx.faz.features.caliber import Caliber
from vascx.faz.features.faz_params import FazParameter, FazParameterType
from vascx.faz.features.tortuosity import LengthMeasure, Tortuosity
from vascx.faz.features.vascular_densities import VascularDensity
from vascx.shared.aggregators import median, std
from vascx.shared.features import FeatureSet

ETDRS_SPEC = ETDRSGridSpecification()
VD_INNER_FIELD = GridFieldSpecification(ETDRS_SPEC, ETDRSRing.Inner)
VD_OUTER_FIELD = GridFieldSpecification(ETDRS_SPEC, ETDRSRing.Outer)
VD_SUP_FIELD = GridFieldSpecification(ETDRS_SPEC, ETDRSQuadrant.Superior)
VD_INF_FIELD = GridFieldSpecification(ETDRS_SPEC, ETDRSQuadrant.Inferior)
VD_LEFT_FIELD = GridFieldSpecification(ETDRS_SPEC, ETDRSQuadrant.Left)
VD_RIGHT_FIELD = GridFieldSpecification(ETDRS_SPEC, ETDRSQuadrant.Right)

basic_features = FeatureSet(
    "basic",
    [
        VascularDensity(),
        VascularDensity(VD_INNER_FIELD),
        VascularDensity(VD_OUTER_FIELD),
        VascularDensity(VD_SUP_FIELD),
        VascularDensity(VD_INF_FIELD),
        VascularDensity(VD_LEFT_FIELD),
        VascularDensity(VD_RIGHT_FIELD),
        FazParameter(FazParameterType.PerimeterLength),
        FazParameter(FazParameterType.Area),
        Caliber(min_numpoints=5, aggregator=median),
        Caliber(min_numpoints=5, aggregator=std),
        Tortuosity(
            length_measure=LengthMeasure.Splines, min_numpoints=5, aggregator=median
        ),
        BifurcationCount(),
    ],
)
