import traceback
import warnings
from typing import Dict, List

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from rtnls_enface.base import EnfaceImage
from rtnls_fundusprep.cfi_bounds import CFIBounds as Bounds
from vascx.faz.feature_sets.basic import *  # noqa: F401
from vascx.fundus.feature_sets.bergmann import *  # noqa: F401
from vascx.fundus.feature_sets.full import *  # noqa: F401
from vascx.fundus.feature_sets.quality import *  # noqa: F401
from vascx.fundus.feature_sets.tortuosity import *  # noqa: F401
from vascx.fundus.retina import Retina
from vascx.shared.features import FeatureSet


def extract_one(
    ex, feature_set_name, retina_cls: EnfaceImage = Retina, print_stack_trace=False
):
    feature_set = FeatureSet.get_by_name(feature_set_name)
    if feature_set is None:
        raise ValueError(f"Feature set '{feature_set_name}' not found.")
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            # add bounds to item
            if "bounds" in ex:
                # print(type(ex["metadata"]), ex["metadata"])
                bounds = Bounds(**ex["bounds"])
                M = bounds.get_cropping_transform(1024)
                ex["bounds"] = bounds.warp(M)
            retina = retina_cls.from_file(**ex)
            features = retina.calc_features(feature_set)

            warnings.simplefilter("always")

            # Modify the warning message with additional information
            warning_messages = [
                f" Example {ex['id']}:: " + str(warning.message)
                for warning in caught_warnings
            ]
            for w in warning_messages:
                print(w)

        return features, warning_messages
    except Exception as err:
        print(f"Exception when computing features for example {str(ex['id'])}")
        traceback.print_exc()
        if print_stack_trace:
            traceback.print_exc()
        else:
            print(err)
        return {}, [f"Error computing features for example {str(ex)}"]


def extract_in_parallel(
    examples: List[Dict],
    feature_set_name: str,
    retina_cls=Retina,
    n_jobs: int = 8,
    print_stack_trace=False,
    logger=None,
):
    res = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(extract_one)(ex, feature_set_name, retina_cls, print_stack_trace)
        for ex in tqdm(examples)
    )

    features = [r[0] for r in res]
    warnings = [w for r in res for w in r[1]]
    if logger is not None:
        for w in warnings:
            logger.warning(w)

    df = pd.DataFrame(features)
    if "id" in examples[0]:
        df.index = [ex["id"] for ex in examples]
    return df
