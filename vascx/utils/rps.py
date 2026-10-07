# BSD 2-Clause License
#
# Copyright (c) 2023, uw-biomedical-ml
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Adapted from https://github.com/uw-biomedical-ml/retinal-pigmentation-score
# Upstream revision: c3404ecdf9368a4e0621ef51e6bdcc528ec2998e

"""Dataset-level PCA postprocessing for the retinal pigmentation score."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def calculate_rps(
    df: pd.DataFrame,
    a_column: str,
    b_column: str,
    *,
    output_column: str = "rps",
) -> pd.DataFrame:
    """Return a copy of a VascX biomarker table with a fitted RPS column.

    Fit a two-component PCA to the finite per-image median a* and b* values,
    without standardizing them. RPS is the centered first-component score.
    As in the original implementation, flip the first component if both of
    its loadings are positive. L* is deliberately not used.

    This fits on the supplied dataset: scores from independently fitted
    datasets/batches are not necessarily comparable. Apply any image-quality
    filtering before calling this function. Missing/nonfinite a* or b* yields
    NaN RPS, preserving row order and index (including duplicate indices).
    At least two complete rows with some colour variation are required.
    Existing output columns are rejected to avoid accidental overwrites.
    """
    if not df.columns.is_unique:
        raise ValueError("The input dataframe must have unique column names")
    if a_column == b_column:
        raise ValueError("a_column and b_column must be different")
    if output_column in df.columns:
        raise ValueError(f"Output column {output_column!r} already exists")
    values = df.loc[:, [a_column, b_column]].to_numpy(dtype=float, na_value=np.nan)
    valid = np.isfinite(values).all(axis=1)
    complete = values[valid]
    if len(complete) < 2:
        raise ValueError("RPS requires at least two rows with finite a* and b* values")
    if np.all(complete == complete[0]):
        raise ValueError("RPS requires variation in a* or b* values")

    pca = PCA(n_components=2, svd_solver="full").fit(complete)
    if np.all(pca.components_[0] > 0):
        pca.components_[0] *= -1
    scores = np.full(len(df), np.nan)
    scores[valid] = pca.transform(complete)[:, 0]
    result = df.copy()
    result[output_column] = scores
    return result
