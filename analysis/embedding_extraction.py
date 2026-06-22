"""
Extract CLIP image and text embeddings for all NSD stimuli.

Image embeddings are obtained from the CLIP ViT-B/32 image encoder.
Text embeddings are obtained by averaging CLIP embeddings across all
available captions associated with each NSD image.

Outputs
-------
nsd_clip_embeddings.npy
nsd_clip_embeddings_text.npy
"""

import warnings
import sys

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import clip
from PIL import Image
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

stimDir = Path("/your/nsd/stimuli/path/")


# -----------------------------
# Extract image embeddings
# -----------------------------
image_embeddings = []

with h5py.File(stimDir / "nsd_stimuli.hdf5", "r") as f:
    img_brick = f["imgBrick"]
    n_images = img_brick.shape[0]

    for i in tqdm(range(n_images), desc="Extracting image embeddings"):
        img_data = img_brick[i, :, :, :]
        img = Image.fromarray(img_data.astype(np.uint8), mode="RGB")

        image_input = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_embeddings.append(image_features.cpu())

image_embeddings = torch.cat(image_embeddings, dim=0)
print("Image embeddings shape:", image_embeddings.shape)

np.save("nsd_clip_embeddings.npy", image_embeddings.numpy())


# -----------------------------
# Extract text embeddings
# -----------------------------
nsd_cap = pd.read_csv(stimDir / "nsd_caption.csv")

text_embeddings = []

for nsd_id in tqdm(range(1, n_images + 1), desc="Extracting text embeddings"):
    captions = nsd_cap.loc[nsd_cap["nsdId"] == nsd_id, "caption"].tolist()

    if len(captions) == 0:
        raise ValueError(f"No captions found for nsdId={nsd_id}")

    tokenized_text = clip.tokenize(captions, truncate=True).to(device)

    with torch.no_grad():
        text_features = model.encode_text(tokenized_text)

    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    mean_text_features = text_features.mean(dim=0)
    mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)

    text_embeddings.append(mean_text_features.cpu())

text_embeddings = torch.stack(text_embeddings, dim=0)
print("Text embeddings shape:", text_embeddings.shape)

np.save("nsd_clip_embeddings_text.npy", text_embeddings.numpy())