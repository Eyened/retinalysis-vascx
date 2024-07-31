from vascx.fundus.features.base import FeatureSet
from vascx.fundus.features.tortuosity import LengthMeasure, Tortuosity
from vascx.shared.aggregators import median

tortuosity_set = FeatureSet(
    {
        "tort": Tortuosity(length_measure=LengthMeasure.Skeleton, aggregator=median),
        "tort_splines": Tortuosity(
            length_measure=LengthMeasure.Splines, aggregator=median
        ),
    }
)
