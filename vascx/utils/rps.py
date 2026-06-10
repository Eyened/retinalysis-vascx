from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def fit_rps_pca(ab: np.ndarray, n_components: int = 2) -> PCA:
    """Fit PCA on (a, b) chromaticity; flip PC1 sign if both loadings are positive."""
    pca = PCA(n_components=n_components)
    pca.fit(ab)

    if (pca.components_[0][0] > 0) and (pca.components_[0][1] > 0):
        pca.components_[0] = -pca.components_[0]

    return pca


def compute_rps(
    df: pd.DataFrame,
    a_col: str,
    b_col: str,
    n_components: int = 2,
) -> pd.Series:
    """Transform (a, b) columns to PC1 RPS scores; NaN where a or b is missing."""
    valid = df[[a_col, b_col]].dropna()
    if valid.empty:
        return pd.Series(np.nan, index=df.index, name="pigmentation")

    pca = fit_rps_pca(valid[[a_col, b_col]].to_numpy(), n_components=n_components)
    scores = pd.Series(np.nan, index=df.index, name="pigmentation")
    transformed = pca.transform(valid[[a_col, b_col]].to_numpy())[:, 0]
    scores.loc[valid.index] = transformed
    return scores
