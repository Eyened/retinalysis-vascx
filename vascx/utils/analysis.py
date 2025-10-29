import traceback
import warnings
from typing import Dict, List, Optional

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from rtnls_enface.base import EnfaceImage
from rtnls_fundusprep.cfi_bounds import CFIBounds as Bounds
from vascx.fundus.feature_sets import *
from vascx.fundus.retina import Retina
from vascx.shared.features import FeatureSet


def extract_one(
    ex,
    feature_set_name,
    retina_cls: EnfaceImage = Retina,
    print_stack_trace: bool = False,
    plots_folder: Optional[str] = None,
):
    feature_set = FeatureSet.get_by_name(feature_set_name)
    if feature_set is None:
        raise ValueError(f"Feature set '{feature_set_name}' not found.")
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            # add bounds to item
            if "bounds" in ex:
                bounds = ex["bounds"]
                if isinstance(bounds, dict):
                    bounds = Bounds(**bounds)
                assert isinstance(bounds, Bounds)
                M = bounds.get_cropping_transform(1024)
                ex["bounds"] = bounds.warp(M)
            retina = retina_cls.from_file(**ex)
            features = retina.calc_features(feature_set, plots_folder)

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
    print_stack_trace: bool = False,
    logger=None,
    plots_folder: Optional[str] = None,
):
    from vascx.shared.features import FeatureSet

    feature_set = FeatureSet.get_by_name(feature_set_name)
    if feature_set is None:
        raise ValueError(f"Feature set '{feature_set_name}' not found.")
    
    #
    
    res = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(extract_one)(ex, feature_set_name, retina_cls, print_stack_trace, plots_folder)
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
