import traceback
import warnings
from pathlib import Path
from typing import Dict, List

import pandas as pd
from joblib import Parallel, delayed
from rtnls_fundusprep.mask_extraction import Bounds
from tqdm import tqdm

from rtnls_enface.base import EnfaceImage
from vascx.faz.feature_sets.basic import *  # noqa: F401
from vascx.fundus.feature_sets.bergmann import *  # noqa: F401
from vascx.fundus.feature_sets.tortuosity import *  # noqa: F401
from vascx.fundus.feature_sets.full import *  # noqa: F401

from vascx.fundus.retina import Retina
from vascx.shared.features import FeatureSet


def extract_one(ex, feature_set_name, retina_cls: EnfaceImage = Retina):
    feature_set = FeatureSet.get_by_name(feature_set_name)
    if feature_set is None:
        raise ValueError(f"Feature set '{feature_set_name}' not found.")
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            # add bounds to item
            if "metadata" in ex:
                bounds = Bounds(**ex["metadata"])
                M = bounds.get_cropping_matrix(1024)
                ex["bounds"] = bounds.warp(M, (1024, 1024))
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
    except Exception:
        traceback.print_exc()
        return {}, [f"Error computing features for example {str(ex)}"]
    
    


def extract_in_parallel(
    examples: List[Dict],
    feature_set_name: str,
    retina_cls=Retina,
    n_jobs: int = 8,
    logger=None,
):
    res = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(extract_one)(ex, feature_set_name, retina_cls) for ex in tqdm(examples)
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
