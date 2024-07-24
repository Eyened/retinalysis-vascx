from vascx.analysis.aggregators import median
from vascx.features.base import FeatureSet
from vascx.features.tortuosity import LengthMeasure, Tortuosity

tortuosity_set = FeatureSet(
    {
        "tort": Tortuosity(length_measure=LengthMeasure.Skeleton, aggregator=median),
        "tort_splines": Tortuosity(
            length_measure=LengthMeasure.Splines, aggregator=median
        ),
    }
)
