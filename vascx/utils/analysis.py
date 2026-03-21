import warnings
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd
from joblib import Parallel, delayed
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
    # try:
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        if "bounds" in ex:
            bounds = ex["bounds"]
            if isinstance(bounds, dict):
                bounds = Bounds(**bounds)
            assert isinstance(bounds, Bounds)
            M = bounds.get_cropping_transform(1024)
            ex["bounds"] = bounds.warp(M)
        retina = retina_cls.from_file(**ex)
        features = retina.calc_features(feature_set, plots_folder)

        warning_messages = [str(w.message) for w in caught_warnings]

    return features, warning_messages
    # except Exception as err:
    #     print(f"Exception when computing features for example {str(ex['id'])}")
    #     if print_stack_trace:
    #         traceback.print_exc()
    #     else:
    #         print(err)
    #     return {}, [f"Error computing features for example {str(ex)}"]


def extract_multiple(
    examples: List[Dict],
    feature_set_name: str,
    retina_cls: EnfaceImage = Retina,
    print_stack_trace: bool = False,
    plots_folder: Optional[str] = None,
):
    """Extract features for multiple examples sequentially."""
    return [
        extract_one(ex, feature_set_name, retina_cls, print_stack_trace, plots_folder)
        for ex in examples
    ]


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

    if len(examples) == 0:
        return pd.DataFrame()

    n_workers = min(n_jobs, len(examples))
    base_batch_size, remainder = divmod(len(examples), n_workers)
    example_batches: List[List[Dict]] = []
    start = 0
    for worker_idx in range(n_workers):
        batch_size = base_batch_size + (1 if worker_idx < remainder else 0)
        end = start + batch_size
        example_batches.append(examples[start:end])
        start = end

    batch_results = Parallel(n_jobs=n_workers, verbose=0)(
        delayed(extract_multiple)(
            batch, feature_set_name, retina_cls, print_stack_trace, plots_folder
        )
        for batch in example_batches
    )
    res = [item for batch in batch_results for item in batch]

    features = [r[0] for r in res]

    warning_counts: Dict[str, int] = defaultdict(int)
    for r in res:
        for w in r[1]:
            warning_counts[w] += 1

    if logger is not None:
        for msg, count in warning_counts.items():
            logger.warning(f"{msg} (x{count})" if count > 1 else msg)
    # else:
    #     for msg, count in warning_counts.items():
    #         suffix = f" (x{count})" if count > 1 else ""
    #         print(f"Warning: {msg}{suffix}")

    df = pd.DataFrame(features)
    if examples and "id" in examples[0]:
        df.index = [ex["id"] for ex in examples]
    return df
