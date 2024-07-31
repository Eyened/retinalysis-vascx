import traceback
import warnings
from pathlib import Path
from typing import Dict, List

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from vascx.fundus.feature_sets import feature_sets
from vascx.fundus.retina import Retina


def make_retina(ex) -> Retina:
    return Retina.from_file(
        ex["av_path"],
        disc_path=ex["disc_path"] if "disc_path" in ex else None,
        fundus_path=ex["fundus_path"] if "fundus_path" in ex else None,
        fovea_location=ex["fovea_location"] if "fovea_location" in ex else None,
        scaling_factor=ex["scale"] if "scale" in ex else 1.0,
    )


def extract_one(ex, feature_set_name):
    feature_set = feature_sets[feature_set_name]
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            retina = make_retina(ex)
            features = retina.calc_features(feature_set)

            warnings.simplefilter("always")

            # Modify the warning message with additional information
            warning_messages = [
                str(warning.message) + f" Warning at example {Path(ex['av_path']).stem}"
                for warning in caught_warnings
            ]

        return features, warning_messages
    except Exception:
        traceback.print_exc()
        return {}, [f"Error computing features for example {str(ex)}"]


def extract_in_parallel(
    examples: List[Dict], feature_set_name: str, n_jobs: int = 8, logger=None
):
    res = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(extract_one)(ex, feature_set_name=feature_set_name)
        for ex in tqdm(examples)
    )

    features = [r[0] for r in res]
    warnings = [w for r in res for w in r[1]]
    if logger is not None:
        for w in warnings:
            logger.warning(w)

    return pd.DataFrame(features)
