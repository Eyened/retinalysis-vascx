from vascx.fundus.features.tortuosity import LengthMeasure, Tortuosity
from vascx.shared.aggregators import median
from vascx.shared.features import FeatureSet

tortuosity_set = FeatureSet(
    "tortuosity",
    {
        "tort": Tortuosity(length_measure=LengthMeasure.Skeleton, aggregator=median),
        "tort_splines": Tortuosity(
            length_measure=LengthMeasure.Splines, aggregator=median
        ),
    },
)
