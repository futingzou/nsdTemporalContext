"""Utility functions for NSD temporal context reinstatement analyses.

This contains reusable functions for loading NSD beta-series data,
constructing HCP-MMP ROI masks, fitting an inverted encoding model with ridge
regression, and evaluating reconstruction accuracy with permutation tests.

"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, Sequence

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

Hemisphere = Literal["lh", "rh", "bilateral"]
SimilarityMetric = Literal["cosine", "pearson"]


ROI_INDEX_MAP: dict[str, list[int]] = {
    "AG": [143, 150, 151],
    "IPS": [144, 145, 146, 17, 117, 95, 48, 50],
    "VMPFC": [65, 88],
    "DLPFC": [73, 83],
    "MPC": [30, 31, 27, 34, 33, 38, 32],
    "PCC": [34, 33, 38, 32, 35, 161, 162],
    "PCUN": [30, 27],
    "LOTC": [158, 20, 21, 159, 156, 23, 2, 157, 138],
    "M1": [8],
    "V1": [1],
    "RSC": [14],
}


DEFAULT_ALPHA_GRID: tuple[float, ...] = (
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
    10000.0,
    100000.0,
)


def get_roi_index_list(roi_name: str) -> list[int]:
    """Return HCP-MMP parcel indices for a named ROI.

    Parameters
    ----------
    roi_name : str
        ROI label. Supported labels are the keys of ``ROI_INDEX_MAP``.

    Returns
    -------
    list[int]
        HCP-MMP atlas indices assigned to the requested ROI.

    Raises
    ------
    ValueError
        If ``roi_name`` is not included in ``ROI_INDEX_MAP``.
    """
    roi_name = roi_name.upper()
    try:
        return ROI_INDEX_MAP[roi_name]
    except KeyError as exc:
        valid = ", ".join(sorted(ROI_INDEX_MAP))
        raise ValueError(f"Unknown ROI '{roi_name}'. Valid ROI names: {valid}.") from exc


def add_adjacent_trial_columns(
    df: pd.DataFrame,
    time_point: int,
    trial_column: str = "FIRST_TRIAL_IN_RUN",
    minus_column: str = "TRIAL_M",
    plus_column: str = "TRIAL_P",
) -> pd.DataFrame:
    """Add columns indexing trials before and after each target trial.

    Parameters
    ----------
    df : pandas.DataFrame
        Behavioral table containing trial numbers.
    time_point : int
        Distance from the target item, in trials.
    trial_column : str, default="FIRST_TRIAL_IN_RUN"
        Column containing target-trial position within run.
    minus_column : str, default="TRIAL_M"
        Name of output column for the preceding trial.
    plus_column : str, default="TRIAL_P"
        Name of output column for the following trial.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with adjacent-trial columns added.
    """
    out = df.copy()
    out[minus_column] = out[trial_column] - time_point
    out[plus_column] = out[trial_column] + time_point
    return out


def _load_hcp_mmp_array(atlas_dir: Path, hemisphere: Literal["lh", "rh"]) -> np.ndarray:
    atlas_file = atlas_dir / "label" / f"{hemisphere}.HCP_MMP1.mgz"
    if not atlas_file.exists():
        raise FileNotFoundError(f"Atlas file not found: {atlas_file}")
    return np.squeeze(nib.load(str(atlas_file)).get_fdata().astype("int16"))


def _make_mask(atlas_array: np.ndarray, roi_indices: Sequence[int]) -> np.ndarray:
    return np.isin(atlas_array, roi_indices)


def gen_roi_mask(
    atlas_dir: str | Path,
    hemisphere: Hemisphere,
    roi_indices: Sequence[int],
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Generate an HCP-MMP ROI mask for one hemisphere or both hemispheres.

    Parameters
    ----------
    atlas_dir : str or pathlib.Path
        Directory containing ``label/lh.HCP_MMP1.mgz`` and/or
        ``label/rh.HCP_MMP1.mgz``.
    hemisphere : {"lh", "rh", "bilateral"}
        Hemisphere to mask.
    roi_indices : sequence of int
        HCP-MMP parcel indices to include in the mask.

    Returns
    -------
    numpy.ndarray or tuple[numpy.ndarray, numpy.ndarray]
        Boolean ROI mask. For ``hemisphere="bilateral"``, returns
        ``(lh_mask, rh_mask)``.
    """
    atlas_dir = Path(atlas_dir)

    if hemisphere == "bilateral":
        lh_mask = _make_mask(_load_hcp_mmp_array(atlas_dir, "lh"), roi_indices)
        rh_mask = _make_mask(_load_hcp_mmp_array(atlas_dir, "rh"), roi_indices)
        return lh_mask, rh_mask

    if hemisphere in {"lh", "rh"}:
        return _make_mask(_load_hcp_mmp_array(atlas_dir, hemisphere), roi_indices)

    raise ValueError("hemisphere must be one of {'lh', 'rh', 'bilateral'}.")


def read_all_beta(
    subject_id: int,
    beta_dir: str | Path,
    space: str,
    hemisphere: Hemisphere,
    scale: float = 300.0,
    dtype: str = "float32",
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Load all beta sessions for one NSD subject.

    Parameters
    ----------
    subject_id : int
        NSD subject number, e.g. ``1`` for ``subj01``.
    beta_dir : str or pathlib.Path
        Root directory containing ``subjXX/<space>/betas_fithrf``.
    space : str
        NSD surface/volume space, e.g. ``"fsaverage"``.
    hemisphere : {"lh", "rh", "bilateral"}
        Hemisphere(s) to load.
    scale : float, default=300.0
        Divisor applied to loaded beta values, matching NSD beta scaling.
    dtype : str, default="float32"
        Numeric dtype used after loading.

    Returns
    -------
    numpy.ndarray or tuple[numpy.ndarray, numpy.ndarray]
        Concatenated beta matrix with shape ``(trials, vertices)``. For
        bilateral loading, returns ``(lh_betas, rh_betas)``.
    """
    beta_dir = Path(beta_dir)
    subj_beta_dir = beta_dir / f"subj{subject_id:02d}" / space / "betas_fithrf"

    def load_hemi(hemi: Literal["lh", "rh"]) -> np.ndarray:
        session_files = sorted(subj_beta_dir.glob(f"{hemi}.betas_session*.mgh"))
        if not session_files:
            raise FileNotFoundError(f"No beta files found for {hemi} in {subj_beta_dir}")

        sessions = [
            np.squeeze(nib.load(str(path)).get_fdata().astype(dtype) / scale).T
            for path in session_files
        ]
        return np.concatenate(sessions, axis=0)

    if hemisphere == "bilateral":
        return load_hemi("lh"), load_hemi("rh")

    if hemisphere in {"lh", "rh"}:
        return load_hemi(hemisphere)

    raise ValueError("hemisphere must be one of {'lh', 'rh', 'bilateral'}.")


def optimal_alpha(
    x: np.ndarray,
    y: np.ndarray,
    alpha_grid: Sequence[float] = DEFAULT_ALPHA_GRID,
    cv: int = 5,
    scoring: str = "neg_mean_squared_error",
    n_jobs: int | None = None,
) -> float:
    """Select the ridge regularization parameter by cross-validation.

    Parameters
    ----------
    x : numpy.ndarray
        Predictor matrix with shape ``(samples, features)``.
    y : numpy.ndarray
        Target matrix with shape ``(samples, targets)``.
    alpha_grid : sequence of float, default=DEFAULT_ALPHA_GRID
        Candidate ridge ``alpha`` values.
    cv : int, default=5
        Number of cross-validation folds.
    scoring : str, default="neg_mean_squared_error"
        Scikit-learn scoring rule.
    n_jobs : int or None, default=None
        Number of parallel jobs used by ``GridSearchCV``.

    Returns
    -------
    float
        Best ridge ``alpha`` value.
    """
    grid_search = GridSearchCV(
        estimator=Ridge(),
        param_grid={"alpha": list(alpha_grid)},
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
    )
    grid_search.fit(x, y)
    return float(grid_search.best_params_["alpha"])


class RidgeRegScratch:
    """Closed-form multi-output ridge regression with reconstruction support.

    This class fits the encoding model ``Y = X B`` and reconstructs latent
    predictors from observed responses via the pseudo-inverse of the fitted
    coefficient matrix. It mirrors the behavior of the original analysis code
    while adding input checks and stable naming.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.thetas_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeRegScratch":
        """Fit a closed-form ridge model.

        Parameters
        ----------
        x : numpy.ndarray
            Predictor matrix with shape ``(samples, features)``.
        y : numpy.ndarray
            Target matrix with shape ``(samples, targets)``.

        Returns
        -------
        RidgeRegScratch
            Fitted estimator.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of samples.")

        x_with_intercept = np.c_[np.ones((x.shape[0], 1)), x]
        penalty = np.eye(x_with_intercept.shape[1])
        penalty[0, 0] = 0.0

        lhs = x_with_intercept.T @ x_with_intercept + self.alpha * penalty
        rhs = x_with_intercept.T @ y
        self.thetas_ = np.linalg.solve(lhs, rhs)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict target responses for new predictors."""
        if self.thetas_ is None:
            raise RuntimeError("Model must be fitted before calling predict().")
        x = np.asarray(x)
        x_with_intercept = np.c_[np.ones((x.shape[0], 1)), x]
        return x_with_intercept @ self.thetas_

    def reconstruct(self, y: np.ndarray) -> np.ndarray:
        """Reconstruct predictor-space coordinates from target responses."""
        if self.thetas_ is None:
            raise RuntimeError("Model must be fitted before calling reconstruct().")
        y = np.asarray(y)
        x_with_intercept = y @ np.linalg.pinv(self.thetas_)
        return x_with_intercept[:, 1:]


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Compute cosine similarity between two arrays after flattening.

    Parameters
    ----------
    a, b : numpy.ndarray
        Arrays to compare. They must contain the same number of elements.
    eps : float, default=1e-12
        Small value used to guard against division by zero.

    Returns
    -------
    float
        Cosine similarity.
    """
    vector_a = np.ravel(a)
    vector_b = np.ravel(b)
    if vector_a.shape != vector_b.shape:
        raise ValueError("a and b must have the same number of elements.")

    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denominator < eps:
        return np.nan
    return float(np.dot(vector_a, vector_b) / denominator)


def rowwise_similarity(
    actual: np.ndarray,
    predicted: np.ndarray,
    metric: SimilarityMetric = "cosine",
) -> np.ndarray:
    """Compute row-wise similarity between actual and predicted matrices."""
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must have identical shapes.")

    if metric == "cosine":
        return np.array([
            cosine_similarity(row_actual, row_predicted)
            for row_actual, row_predicted in zip(actual, predicted)
        ])

    if metric == "pearson":
        values = []
        for row_actual, row_predicted in zip(actual, predicted):
            r = np.corrcoef(row_actual, row_predicted)[0, 1]
            values.append(np.arctanh(r))
        return np.array(values)

    raise ValueError("metric must be either 'cosine' or 'pearson'.")


def sim_permutation(
    actual: np.ndarray,
    predicted: np.ndarray,
    metric: SimilarityMetric,
    roi_name: str,
    subject_id: int,
    n_permutations: int = 1000,
    random_state: int | np.random.Generator | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Estimate reconstruction similarity against a permutation null.

    Parameters
    ----------
    actual : numpy.ndarray
        Ground-truth embedding matrix with shape ``(items, features)``.
    predicted : numpy.ndarray
        Reconstructed embedding matrix with shape ``(items, features)``.
    metric : {"cosine", "pearson"}
        Similarity metric. Pearson correlations are Fisher-z transformed before
        averaging, matching the original analysis.
    roi_name : str
        ROI label to include in the output table.
    subject_id : int
        Subject identifier to include in the output table.
    n_permutations : int, default=1000
        Number of label permutations used to estimate the null distribution.
    random_state : int, numpy.random.Generator, or None, default=None
        Seed or generator for reproducible permutations.
    verbose : bool, default=True
        Whether to print summary statistics.

    Returns
    -------
    pandas.DataFrame
        One-row table containing subject, ROI, observed similarity, null mean,
        z-score, and one-sided permutation p-value.
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must have identical shapes.")

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

    observed = float(np.nanmean(rowwise_similarity(actual, predicted, metric)))
    null_values = np.empty(n_permutations, dtype=float)

    for idx in range(n_permutations):
        permuted = predicted[rng.permutation(predicted.shape[0])]
        null_values[idx] = np.nanmean(rowwise_similarity(actual, permuted, metric))

    null_mean = float(np.nanmean(null_values))
    null_std = float(np.nanstd(null_values, ddof=0))
    z_score = np.nan if null_std == 0 else float((observed - null_mean) / null_std)
    p_value = float((np.sum(null_values >= observed) + 1) / (n_permutations + 1))

    if verbose:
        print(f"subj={subject_id}, roi={roi_name}")
        print(f"averaged similarity = {observed}")
        print(f"null similarity = {null_mean}")
        print(f"z={z_score}, p={p_value}")

    return pd.DataFrame(
        {
            "subj": subject_id,
            "roi": roi_name,
            "mean_sim": observed,
            "null_sim": null_mean,
            "z_score": z_score,
            "pval": p_value,
        },
        index=[0],
    )


def extract_bilateral_roi_betas(
    beta_data: tuple[np.ndarray, np.ndarray],
    roi_mask: tuple[np.ndarray, np.ndarray],
    drop_nan_columns: bool = True,
) -> np.ndarray:
    """Apply bilateral ROI masks and concatenate left/right beta matrices.

    Parameters
    ----------
    beta_data : tuple[numpy.ndarray, numpy.ndarray]
        ``(lh_betas, rh_betas)`` returned by ``read_all_beta(..., 'bilateral')``.
    roi_mask : tuple[numpy.ndarray, numpy.ndarray]
        ``(lh_mask, rh_mask)`` returned by ``gen_roi_mask(..., 'bilateral')``.
    drop_nan_columns : bool, default=True
        Whether to remove vertices containing any NaN values.

    Returns
    -------
    numpy.ndarray
        ROI beta matrix with left and right hemisphere vertices concatenated.
    """
    lh_betas, rh_betas = beta_data
    lh_mask, rh_mask = roi_mask
    roi_betas = np.concatenate([lh_betas[:, lh_mask], rh_betas[:, rh_mask]], axis=1)

    if drop_nan_columns:
        roi_betas = roi_betas[:, ~np.isnan(roi_betas).any(axis=0)]

    return roi_betas


def nsd_trial_indices(
    session: Iterable[int],
    trial: Iterable[int],
    trials_per_session: int = 750,
) -> np.ndarray:
    """Convert NSD one-indexed session/trial columns to zero-indexed row indices."""
    session = np.asarray(session, dtype=np.int64)
    trial = np.asarray(trial, dtype=np.int64)
    return (session - 1) * trials_per_session + trial - 1

