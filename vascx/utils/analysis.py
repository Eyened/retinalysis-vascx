import traceback
import warnings
from pathlib import Path
from typing import Dict, List

import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from vascx.feature_sets import feature_sets
from vascx.retina import Retina


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


def match_cases_controls(df, covariates, treatment):
    # Assuming you have a DataFrame `df` with a binary treatment indicator 'treatment' and covariates 'X'
    X = df[covariates]  # covariates
    y = df[treatment]  # treatment indicator

    # Logistic regression to compute propensity scores
    logit_model = LogisticRegression()
    logit_model.fit(X, y)
    propensity_scores = logit_model.predict_proba(X)[:, 1]

    # Create two datasets for treatment and control
    cases = df[df[treatment] == 1]
    ctrls = df[df[treatment] == 0]

    # Fit a nearest neighbors model on control group
    nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(ctrls[covariates])

    # For each treated subject, find the closest control subject
    distances, indices = nn.kneighbors(cases[covariates])

    # Create a DataFrame with matched pairs
    matched_pairs = cases.assign(match=indices.squeeze())
    return df
