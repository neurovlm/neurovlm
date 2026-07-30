# Mapping Brain Networks to Cognitive Atlas Concepts

This tutorial shows how to use **NeuroVLM** to map a brain network to the most relevant cognitive functions using the Cognitive Atlas dataset.

The workflow:

1. Load a brain network
2. Encode the brain map with the NeuroVLM autoencoder
3. Compare it with CogAtlas concept embeddings
4. Retrieve the most similar cognitive concepts

---

# Step 1 — Import libraries

```python
import torch
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nilearn.image import resample_img
from neurovlm.data import load_dataset, load_masker, load_latent
from neurovlm.models import load_model
from neurovlm.train import which_device
```

# Step 2 — Load Available Brain Networks

First, load the available functional brain networks included with NeuroVLM.

```python
networks = load_dataset("networks")

print("Available networks:")

for net in networks["Du"].keys():
    print(net)
```

# Step 3 — Select a Network

Choose one of the available networks to analyze, and convert it into a NIfTI brain image

```python
choice = "Default"
arr = networks["Du"][choice]["array"] > 0

img = nib.Nifti1Image(
    arr.astype(float),
    affine=networks["Du"][choice]["affine"]
)
```
This creates a brain image that can be processed by the NeuroVLM model.

# Step 4 — Load NeuroVLM Models

Load the pretrained NeuroVLM models used to project brain images and text into a shared embedding space.

```python
autoencoder = load_model("autoencoder")
proj_head_text = load_model("proj_head_text_infonce")
proj_head_img = load_model("proj_head_image_infonce")
```
These models perform three key roles:
Autoencoder — converts brain images into latent vectors
Text projection head — aligns text embeddings with brain embeddings
Image projection head — aligns brain embeddings with text embeddings
Together, these allow comparisons between brain activation maps and cognitive concepts

# Step 5 — Encode the Brain Image

Next, transform the brain image into the format required by the NeuroVLM model.

```python
masker = load_masker()

img_resampled = resample_img(img, masker.mask_img.affine)

img_tensor = torch.from_numpy(
    masker.transform(img_resampled)
)

img_latent = autoencoder.encoder(img_tensor)
```
This step performs three operations:
1) Resampling the image to match the model’s brain mask
2) Vectorizing the brain image
3) Encoding the brain data into a latent representation
The resulting img_latent vector represents the brain activation pattern in the NeuroVLM embedding space.

# Step 6 — Load Cognitive Atlas Concepts

Next, load embeddings representing **Cognitive Atlas concepts**.

```python
latent_text, concept_terms = load_latent("cogatlas")

df = load_dataset("cogatlas")
```
The Cognitive Atlas is a structured knowledge base of cognitive neuroscience concepts.
Examples include:
- working memory
- attention
- ognitive control
- decision making

# Step 7 — Compute Cosine Similarity

To determine which cognitive concepts are most related to the brain network, compute the cosine similarity between the brain embedding and each concept embedding.

```python
cos_sim = (img_latent_aligned @ latent_text_aligned.T).squeeze()

inds = torch.argsort(cos_sim, descending=True)
```
Cosine similarity measures how closely two vectors align in the shared embedding space.
Higher similarity scores indicate stronger relationships between the brain network and a cognitive concept.

# Step 8 — View Top Concepts

Retrieve the top concepts with the highest similarity scores.

```python
top_k = 10
top_inds = inds[:top_k].cpu().numpy()

top_concepts = df.iloc[top_inds]["term"]
```
Example output:
1 working memory
2 attention
3 cognitive control
4 decision making
These results suggest the cognitive functions most strongly associated with the selected brain network.

# Step 9 — Visualize the Results

Finally, visualize the similarity scores for the top cognitive concepts.

```python
plt.barh(top_concepts, top_scores)
plt.gca().invert_yaxis()
plt.xlabel("Cosine Similarity")
plt.title("Top Cognitive Atlas Concepts")
plt.show()
```
The bar chart displays the cognitive concepts most strongly associated with the brain network.

# Summary

In this tutorial, we:

- Loaded a brain network dataset
- Converted the network into a brain image
- Encoded the brain activation map using NeuroVLM
- Compared the brain representation to Cognitive Atlas concept embeddings
- Retrieved the most relevant cognitive functions

This workflow demonstrates how NeuroVLM can bridge **brain activation patterns and cognitive neuroscience concepts**, enabling interpretable mapping between neuroimaging data and cognitive functions
