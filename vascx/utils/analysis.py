import os
import warnings
from pathlib import Path
from typing import Union, Dict, List

from joblib import Parallel, delayed
from tqdm import tqdm, trange
import pandas as pd
from sklearn.linear_model import LogisticRegression

from vascx.retina import Retina
from vascx.feature_sets import feature_sets
from vascx import Retina


def get_examples_dict(data_path: Union[str, Path]):
    data_path = Path(data_path)

    foveas = pd.read_csv(data_path / "fovea.csv", index_col=0)

    masks_path = data_path / "av"
    discs_path = data_path / "discs"
    fundus_path = data_path / "preprocessed_rgb"

    masks = sorted(list(masks_path.glob("*.png")))
    discs = sorted(list(discs_path.glob("*.png")))
    fundus = sorted(list(fundus_path.glob("*.png")))

    ids_list = [int(m.stem.split("_")[0]) for m in masks]
    discs = [d for d in discs if int(d.stem.split("_")[0]) in ids_list]
    fundus = [f for f in fundus if int(f.stem.split("_")[0]) in ids_list]
    fovea_locs = [
        (foveas.loc[id, "mean_x"], foveas.loc[id, "mean_y"]) for id in ids_list
    ]

    meta_path = data_path / "meta.csv"
    if meta_path.exists():
        meta = pd.read_csv(meta_path, index_col="id")
        if "scaling_w" in meta.columns and "scale" in meta.columns:
            scale_factors = [
                meta.loc[id, "scaling_w"] * meta.loc[id, "scale"] for id in ids_list
            ]
        else:
            scale_factors = [1.0 for id in ids_list]
    else:
        scale_factors = [1.0 for id in ids_list]

    return {
        ex[0]: {
            "scale": ex[1],
            "av": ex[2],
            "disc": ex[3],
            "fundus": ex[4],
            "fovea": ex[5],
        }
        for ex in zip(ids_list, scale_factors, masks, discs, fundus, fovea_locs)
    }


def make_retina(ex) -> Retina:
    return Retina.from_file(
        ex["av"],
        disc_path=ex["disc"] if "disc" in ex else None,
        fundus_path=ex["fundus"] if "fundus" in ex else None,
        fovea_location=ex["fovea"] if "fovea" in ex else None,
        scaling_factor=ex["scale"],
    )


def extract_one(ex, feature_set_name):
    feature_set = feature_sets[feature_set_name]
    try:
        retina = make_retina(ex)
        return retina.calc_features(feature_set)
    except Exception as e:
        raise (e)
        warnings.warn(f"Error computing features for file {ex['av']}")
        return {}


def extract_in_parallel(examples: List[Dict], feature_set_name: str, n_jobs: int = 8):
    return pd.DataFrame(
        Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(extract_one)(ex, feature_set_name=feature_set_name)
            for ex in tqdm(examples)
        )
    )


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
