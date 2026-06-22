"""
NSD temporal context reinstatement analysis.

Example code for an inverted encoding model to reconstruct E1 temporal
context image PCs from E2 brain activity in cortical ROIs defined using
the HCP-MMP atlas

Settings
-----------------
space : fsaverage
atlas : HCP_MMP1

Outputs
-------
- Reconstructed embeddings
- Cosine similarity between reconstructed and target embeddings
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from utils import (
    add_adjacent_trial_columns,
    extract_bilateral_roi_betas,
    gen_roi_mask,
    get_roi_index_list,
    nsd_trial_indices,
    read_all_beta,
    RidgeRegScratch,
    optimal_alpha,
    sim_permutation,
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SUBJECTS = range(1, 9)
ROIS = ['AG', 'LOTC', 'V1', 'M1', "VMPFC", "DLPFC", "IPS", "MPC", "PCC", "PCUN", "RSC"]

SECOND_BEH = 1              # 1 = remembered at E2; 0 = not remembered at E2
TIME_POINT = 1              # adjacent item distance from target
IMAGE_SET = "_within"       # "_within" or "_across"

SPACE = "fsaverage"
HEMISPHERE = "bilateral"
N_COMPONENTS = 20           # number of PCA components

TRAIN_TP = 0                # 0 = train on n-0 item; 1 = train on adjacent items
TP_LIST = ["_M", "_P"]
WEIGHTS = [1, 1]
ALPHA = None                 # set to None to select alpha by grid search CV
N_PERMUTATIONS = 1000
RANDOM_SEED = 12345

BASE_DIR: Path = Path('/your/nsd/path/')
FUNC_DIR = BASE_DIR / "nsddata_betas" / "ppdata"
ROI_DIR = BASE_DIR / "nsddata" / "freesurfer" / "fsaverage"
INFO_DIR = Path('/your/nsd/behav/path/')
EMBEDDING_FILE = Path("nsd_clip_embeddings.npy") # or nsd_clip_embeddings_text.npy for text embeddings
OUT_DIR = Path(".")


# -----------------------------------------------------------------------------
# Behavioral preparation
# -----------------------------------------------------------------------------
def prepare_behavior() -> pd.DataFrame:
    """Load behavioral files, apply trial filters, and attach CLIP embeddings."""
    item_info = pd.read_csv(INFO_DIR / "nsdmain_all.csv")
    item_info_raw = pd.read_csv(INFO_DIR / "rawmain.csv")
    embeddings = np.load(EMBEDDING_FILE)

    # Remove within-run repetitions.
    item_info = item_info[
        ~(
            (item_info["FIRST_SESS"] == item_info["SECOND_SESS"])
            & (item_info["FIRST_RUN"] == item_info["SECOND_RUN"])
        )
    ].reset_index(drop=True)

    # Restrict image set by E1E2 lag.
    if IMAGE_SET == "_within":
        item_info = item_info[
            ~(
                (item_info["FIRST_SESS"] != item_info["SECOND_SESS"])
                & (item_info["LAG1"] >= np.log(60 * 60 * 24))
            )
        ].reset_index(drop=True)
    elif IMAGE_SET == "_across":
        item_info = item_info[
            ~(
                (item_info["FIRST_SESS"] == item_info["SECOND_SESS"])
                & (item_info["LAG1"] < np.log(60 * 60 * 24))
            )
        ].reset_index(drop=True)
    else:
        raise ValueError("IMAGE_SET must be either '_within' or '_across'.")

    # Select clean trials based on E2 behavior.
    item_info = item_info[
        (item_info["FIRST_CORRECT"] == 1)
        & (item_info["SECOND_CORRECT"] == SECOND_BEH)
        & (item_info["FIRST_CHANGE"] == 0)
        & (item_info["SECOND_CHANGE"] == 0)
        & (item_info["LAST_CHANGE"] == 0)
    ].reset_index(drop=True)

    # Remove boundary trials where adjacent items are unreliable (i.e., at the edges of the run).
    item_info = item_info[
        (item_info["FIRST_TRIAL_IN_RUN"] > 3)
        & (item_info["FIRST_TRIAL_IN_RUN"] < 61)
    ].reset_index(drop=True)
    item_info = item_info[
        ~(
            (item_info["FIRST_RUN"] % 2 == 0)
            & (item_info["FIRST_TRIAL_IN_RUN"] >= 60)
        )
    ].reset_index(drop=True)

    df = add_adjacent_trial_columns(item_info, TIME_POINT)

    # Attach neighboring-item NSD IDs.
    for suffix in TP_LIST:
        df = pd.merge(
            df,
            item_info_raw[["SUBJECT", "SESSION", "RUN", "TRIAL", "73KID"]],
            left_on=["SUBJECT", "FIRST_SESS", "FIRST_RUN", f"TRIAL{suffix}"],
            right_on=["SUBJECT", "SESSION", "RUN", "TRIAL"],
            how="left",
            suffixes=("", suffix),
        )

    df = df.dropna(subset=["73KID_M", "73KID_P"]).reset_index(drop=True)

    # NSD 73K IDs are one-indexed; numpy arrays are zero-indexed.
    df["caption"] = [embeddings[int(idx) - 1, :] for idx in df["73KID"].values]
    df["caption_M"] = [embeddings[int(idx) - 1, :] for idx in df["73KID_M"].values]
    df["caption_P"] = [embeddings[int(idx) - 1, :] for idx in df["73KID_P"].values]

    return df


# -----------------------------------------------------------------------------
# Model helpers
# -----------------------------------------------------------------------------
def get_train_test_embeddings(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return PCA-projected train/test embeddings for the configured model."""
    if TRAIN_TP == 0: # train on current item only
        train_raw = np.vstack(df_train["caption"].to_numpy())
        test_raw = np.vstack(df_test["caption"].to_numpy())

        pca = PCA(n_components=N_COMPONENTS)
        pca.fit(train_raw)
        return pca.transform(train_raw), pca.transform(test_raw)

    if TRAIN_TP == 1: # train on both adjacent items
        train_raw_by_tp = [np.vstack(df_train[f"caption{tp}"].to_numpy()) for tp in TP_LIST]
        test_raw_by_tp = [np.vstack(df_test[f"caption{tp}"].to_numpy()) for tp in TP_LIST]

        pca = PCA(n_components=N_COMPONENTS)
        pca.fit(np.vstack(train_raw_by_tp))

        train_by_tp = [pca.transform(x) for x in train_raw_by_tp]
        test_by_tp = [pca.transform(x) for x in test_raw_by_tp]

        train_embeddings = np.average(train_by_tp, axis=0, weights=WEIGHTS)
        test_embeddings = np.average(test_by_tp, axis=0, weights=WEIGHTS)
        return train_embeddings, test_embeddings

    raise ValueError("TRAIN_TP must be 0 or 1.")


def run_subject_roi(
    df_subject: pd.DataFrame,
    roi_betas: np.ndarray,
    subject_id: int,
    roi_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run leave-one-FIRST-session-out reconstruction for one subject and ROI."""
    actual_all = []
    pred_all = []
    nsdid_all = []

    for heldout_session in sorted(df_subject["FIRST_SESS"].unique()):
        df_train = df_subject[df_subject["FIRST_SESS"] != heldout_session].reset_index(drop=True)
        df_test = df_subject[df_subject["FIRST_SESS"] == heldout_session].reset_index(drop=True)

        train_embeddings, test_embeddings = get_train_test_embeddings(df_train, df_test)

        train_idx = nsd_trial_indices(df_train["SECOND_SESS"], df_train["SECOND_TRIAL"])
        test_idx = nsd_trial_indices(df_test["SECOND_SESS"], df_test["SECOND_TRIAL"])

        train_betas = roi_betas[train_idx.astype("int64"), :]
        test_betas = roi_betas[test_idx.astype("int64"), :]

        if ALPHA is None:
            alpha = optimal_alpha(train_embeddings, train_betas)
            print(f"Optimal alpha by CV: {alpha:.4f}")
        else:            
            alpha = ALPHA
        model = RidgeRegScratch(alpha=alpha)
        model.fit(train_embeddings, train_betas)
        pred_embeddings = model.reconstruct(test_betas)

        actual_all.append(test_embeddings)
        pred_all.append(pred_embeddings)
        nsdid_all.extend(df_test["73KID"].tolist())

    actual = np.vstack(actual_all)
    predicted = np.vstack(pred_all)

    embedding_output = pd.DataFrame(
        {
            "subj": subject_id,
            "roi": roi_name,
            "nsdid": nsdid_all,
            "actual_embed": actual.tolist(),
            "pred_embed": predicted.tolist(),
        }
    )

    stats_output = sim_permutation(
        actual=actual,
        predicted=predicted,
        metric="cosine",
        roi_name=roi_name,
        subject_id=subject_id,
        n_permutations=N_PERMUTATIONS,
        random_state=RANDOM_SEED + subject_id,
        verbose=True,
    )

    return stats_output, embedding_output


# -----------------------------------------------------------------------------
# Main analysis
# -----------------------------------------------------------------------------
def main() -> None:
    print(f"time_point={TIME_POINT}, image_set={IMAGE_SET}, second_beh={SECOND_BEH}")
    df_beh = prepare_behavior()

    ridge_outputs = []
    pc_outputs = []

    for subject_id in SUBJECTS:
        print(f"\nSubject {subject_id}")
        df_subject = df_beh[df_beh["SUBJECT"] == subject_id].reset_index(drop=True)
        beta_subject = read_all_beta(subject_id, FUNC_DIR, SPACE, HEMISPHERE)

        for roi_name in ROIS:
            print(f"ROI: {roi_name}")
            roi_indices = get_roi_index_list(roi_name)
            roi_mask = gen_roi_mask(ROI_DIR, HEMISPHERE, roi_indices)
            roi_betas = extract_bilateral_roi_betas(beta_subject, roi_mask)

            stats_df, embeddings_df = run_subject_roi(
                df_subject=df_subject,
                roi_betas=roi_betas,
                subject_id=subject_id,
                roi_name=roi_name,
            )
            ridge_outputs.append(stats_df)
            pc_outputs.append(embeddings_df)

    ridge_output = pd.concat(ridge_outputs, ignore_index=True)
    pc_output = pd.concat(pc_outputs, ignore_index=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ridge_file = OUT_DIR / f"clip_ridge_tp{TIME_POINT}{IMAGE_SET}_beh{SECOND_BEH}_allrois.csv"
    embedding_file = OUT_DIR / f"clip_tp{TIME_POINT}{IMAGE_SET}_beh{SECOND_BEH}_allrois.csv"

    ridge_output.to_csv(ridge_file, index=False, encoding="utf-8-sig")
    pc_output.to_csv(embedding_file, index=False, encoding="utf-8-sig")

    print(f"\nSaved: {ridge_file}")
    print(f"Saved: {embedding_file}")


if __name__ == "__main__":
    main()
