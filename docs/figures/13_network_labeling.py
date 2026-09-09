# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: .env
#     language: python
#     name: python3
# ---

# %%
from tqdm.notebook import tqdm
import pickle, gzip
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch

import nibabel as nib
from nilearn.image import resample_img

from neurovlm.data import data_dir, load_dataset
from neurovlm.models import Specter
from neurovlm.resources.loaders import (
    _load_autoencoder, _proj_head_image_infonce, _proj_head_text_infonce, _load_masker
)

from neurovlm.evaluation.notebook_utils import resolve_evaluation_output_dir

evaluation_output_dir = resolve_evaluation_output_dir()
evaluation_output_dir.mkdir(parents=True, exist_ok=True)

# %%
# Load network atlases
networks = load_dataset("networks")

network_imgs = []
for k in networks.keys():
    for a in networks[k].keys():
        network_imgs.append((k, a, nib.Nifti1Image(networks[k][a]["array"], affine=networks[k][a]["affine"])))

networks = [i for i in network_imgs if i[0] not in ["UKBICA", "HCPICA"]]

# Load models
specter = Specter("allenai/specter2_aug2023refresh", adapter="adhoc_query")
proj_head_text = _proj_head_text_infonce()
proj_head_image = _proj_head_image_infonce()
autoencoder = _load_autoencoder()
masker = _load_masker()

# %%
if not (data_dir / "networks_emb.pt").exists():
    # Resample networks
    networks_resampled = []

    for img in tqdm(networks, total=len(networks)):
        img = img[2]
        img_arr = img.get_fdata()

        if len(np.unique(img_arr)) == 2:
            # binary data
            img_resampled = resample_img(img, masker.affine_, interpolation="nearest")
        else:
            img_resampled = resample_img(img, masker.affine_)
            img_resampled_arr = img_resampled.get_fdata()
            img_resampled_arr[img_resampled_arr < 0] = 0.
            thresh = np.percentile(img_resampled_arr.flatten(), 95)
            img_resampled_arr[img_resampled_arr < thresh] = 0.
            img_resampled_arr[img_resampled_arr >= thresh] = 1.
            img_resampled = nib.Nifti1Image(img_resampled_arr, affine=masker.affine_)

        networks_resampled.append(img_resampled)

    # Encode networks
    networks_embed = []
    for v in tqdm(networks_resampled, total=len(networks_resampled)):
        with torch.no_grad():
            networks_embed.append(autoencoder.encoder(torch.from_numpy(masker.transform(v))))
    networks_embed = torch.vstack(networks_embed)
    torch.save(networks_embed, data_dir / "networks_emb.pt")

else:
    networks_embed = torch.load(data_dir / "networks_emb.pt")

# %%
label_map = [
 ('VIS-P', 'unknown'), # this is a precuneus mask
 ('CG-OP', 'cingulo_opercular'),
 ('DN-B', 'default_mode'),
 ('SMOT-B', 'motor'),
 ('AUD', 'auditory'),
 ('PM-PPr', 'motor'),
 ('dATN-B', 'attention'),
 ('SMOT-A', 'motor'),
 ('LANG', 'language'),
 ('FPN-B', 'frontoparietal_control'),
 ('FPN-A', 'frontoparietal_control'),
 ('dATN-A', 'attention'),
 ('VIS-C', 'visual'),
 ('SAL/PMN', 'cingulo_opercular'),
 ('DN-A', 'default_mode'),
 ('NONE', 'unknown'),
 ('Visual1', 'visual'),
 ('Visual2', 'visual'),
 ('Somatomotor', 'motor'),
 ('CingOperc', 'cingulo_opercular'),
 ('DorsAttn', 'attention'),
 ('Language', 'language'),
 ('FrontPar', 'frontoparietal_control'),
 ('Auditory', 'auditory'),
 ('Default', 'default_mode'),
 ('PostMulti', 'unknown'),
 ('VentMulti', 'unknown'),
 ('OrbitAffective', 'unknown'),
 ('Emo/Interoception1', 'unknown'),
 ('Emo/Interoception2', 'unknown'),
 ('Emo/Interoception3', 'unknown'),
 ('Emo/Interoception4', 'unknown'),
 ('Mot/Visspatial1', 'motor'),
 ('Mot/Visspatial2', 'motor'),
 ('Mot/Visspatial3', 'motor'),
 ('Mot/Visspatial4', 'motor'),
 ('Visual1', 'visual'),
 ('Visual2', 'visual'),
 ('Visual3', 'visual'),
 ('DivergentCog1', 'unknown'),
 ('DivergentCog3', 'unknown'),
 ('DivergentCog4', 'unknown'),
 ('DivergentCog5', 'unknown'),
 ('DivergentCog6', 'unknown'),
 ('medial frontal', 'frontoparietal_control'),
 ('frontoparietal', 'frontoparietal_control'),
 ('default mode', 'default_mode'),
 ('motor cortex', 'motor'),
 ('visual A', 'visual'),
 ('visual B', 'visual'),
 ('visual association', 'visual'),
 ('subcortical cerebellum', 'unknown'),
 ('AntSal', 'cingulo_opercular'),
 ('Auditory', 'auditory'),
 ('DorsalDMN', 'default_mode'),
 ('HighVisual', 'visual'),
 ('Language', 'language'),
 ('LECN', 'frontoparietal_control'),
 ('PostSal', 'cingulo_opercular'),
 ('Precuneus', 'default_mode'),
 ('PrimVisual', 'visual'),
 ('RECN', 'frontoparietal_control'),
 ('Sensorimotor', 'motor'),
 ('VentralDMN', 'default_mode'),
 ('Visuospatial', 'unknown'),
 ('Default', 'default_mode'),
 ('LatVis', 'visual'),
 ('FrontPar', 'frontoparietal_control'),
 ('MedVis', 'visual'),
 ('DorsAttn', 'attention'),
 ('Premotor', 'motor'),
 ('Language', 'language'),
 ('Salience', 'cingulo_opercular'),
 ('CingOperc', 'cingulo_opercular'),
 ('HandSM', 'motor'),
 ('FaceSM', 'motor'),
 ('Auditory', 'auditory'),
 ('AntMTL', 'unknown'),
 ('PostMTL', 'unknown'),
 ('ParMemory', 'unknown'),
 ('Context', 'unknown'),
 ('FootSM', 'motor'),
 ('Visual', 'visual'),
 ('VentAttn', 'unknown'), # focus on dorsal attention
 ('DorsalSM', 'motor'),
 ('VentralSM', 'motor'),
 ('MedPar', 'default_mode'),
 ('ParOcc', 'visual'),
 ('SCAN', 'frontoparietal_control'),
 ('Cingulo-Opercular', 'cingulo_opercular'),
 ('Effector-hand', 'motor'),
 ('Effector-mouth', 'motor'),
 ('Effector-foot', 'motor'),
 ('SM', 'motor'),
 ('LateralSM', 'motor'),
 ('ResponseOneHanded(1RESP)', 'motor'),
 ('ResponseTwoHanded(2RESP)', 'motor'),
 ('AuditoryAttentionResponse(AAR)', 'auditory'),
 ('AuditoryPrimarySensory(AUD)', 'auditory'),
 ('DMNNovel(DMNA)', 'default_mode'),
 ('DMNTraditional(DMNB)', 'default_mode'),
 ('FocusOnVisualFeatures(FoVF)', 'visual'),
 ('Initiation(INIT)', 'unknown'),
 ('Language(LN)', 'language'),
 ('MAIN', 'unknown'),
 ('MultipleDemand(MDN)', 'frontoparietal_control'),
 ('Re-evaluation(RE-EV)', 'frontoparietal_control'),
 ('DefaultA', 'default_mode'),
 ('DefaultB', 'default_mode'),
 ('DefaultC', 'default_mode'),
 ('Language', 'language'),
 ('ContA', 'frontoparietal_control'),
 ('ContB', 'frontoparietal_control'),
 ('ContC', 'frontoparietal_control'),
 ('SalVenAttnA', 'cingulo_opercular'),
 ('SalVenAttnB', 'cingulo_opercular'),
 ('DorsAttnA', 'attention'),
 ('DorsAttnB', 'attention'),
 ('Aud', 'auditory'),
 ('SomMotA', 'motor'),
 ('SomMotB', 'motor'),
 ('VisualA', 'visual'),
 ('VisualB', 'visual'),
 ('VisualC', 'visual'),
 ('TempPar', 'frontoparietal_control'),
 ('LimbicA', 'unknown'), # these are dlmpc and temporal lobe
 ('LimbicB', 'unknown'),
 ('SalVentAttnB', 'cingulo_opercular'),
 ('SalVentAttnA', 'cingulo_opercular'),
 ('VisPeri', 'visual'),
 ('VisCent', 'visual'),
 ('SomatomotorA', 'motor'),
 ('SomatomotorB', 'motor'),
 ('Sal/VenAttnA', 'cingulo_opercular'),
 ('Sal/VenAttnB', 'cingulo_opercular'),
 ('ControlC', 'frontoparietal_control'),
 ('ControlA', 'frontoparietal_control'),
 ('ControlB', 'frontoparietal_control'),
 ('Visual', 'visual'),
 ('Somatomotor', 'motor'),
 ('DorsAttn', 'attention'),
 ('Sal/VenAttn', 'cingulo_opercular'),
 ('Limbic', 'unknown'),
 ('Control', 'frontoparietal_control'),
 ('Default', 'default_mode')]


target_labels = [
    "visual",
    "motor",
    "auditory",
    "language",
    "attention",
    "frontoparietal_control",
    "cingulo_opercular",
    "default_mode",
    "limbic_affective",
    "memory_mtl",
    "unknown",
]

labels = [i[1] for i in label_map]
assert len(labels) == 145

# %%
text_labels = {
    "Language": (
        "Language network (LAN; perisylvian language network; frontotemporal language system) [SEP] "
        "Primary regions: left inferior frontal gyrus (Broca’s complex), posterior superior/middle temporal gyrus, "
        "temporoparietal junction/angular gyrus. "
        "Function: speech/text comprehension and production—semantics, syntax, phonology, and sentence-level integration."
    ),
    "Auditory": (
        "Auditory network (AUD; auditory cortex network) [SEP] "
        "Primary regions: Heschl’s gyrus (primary auditory cortex), planum temporale, superior temporal gyrus. "
        "Function: acoustic feature analysis (pitch/timbre/timing), auditory scene analysis, early speech-sound encoding, "
        "and detection of salient sounds."
    ),
    "Default Mode": (
        "Default mode network (DMN; default network; default state network) [SEP] "
        "Primary regions: medial prefrontal cortex, posterior cingulate/precuneus, angular gyrus (lateral parietal). "
        "Function: internally oriented cognition—autobiographical/episodic memory, self-referential thought, future simulation, "
        "social inference, and narrative/semantic integration."
    ),
    "Frontoparietal Control": (
        "Frontoparietal control network (FPCN; frontoparietal network/FPN; central executive network/CEN) [SEP] "
        "Primary regions: dorsolateral/rostrolateral prefrontal cortex (middle frontal), posterior parietal cortex "
        "(inferior parietal/around intraparietal sulcus). "
        "Function: goal maintenance and flexible executive control—working memory, rule implementation, planning, and rapid task switching."
    ),
    "Attention": (
        "Dorsal attention network [SEP] "
        "Primary regions: intraparietal sulcus/superior parietal lobule and frontal eye fields. "
        "Function: goal-directed, voluntary control of visuospatial attention."

    ),
    "Visual": (
        "Visual network (VIS; occipital visual network) [SEP] "
        "Primary regions: calcarine cortex/V1 (cuneus/lingual), extrastriate occipital cortex, ventral occipitotemporal visual areas. "
        "Function: visual feature processing (form/color/motion), object/scene representations, and visuospatial analysis."
    ),
    "Motor": (
        "Motor network (motor/sensorimotor network; SMN) [SEP] "
        "Primary regions: primary motor cortex (precentral gyrus), supplementary motor area, premotor cortex; "
        "with primary somatosensory cortex (postcentral gyrus) for sensorimotor integration. "
        "Function: movement planning, initiation, and execution, plus proprioceptive/tactile feedback used to control actions "
        "and maintain body-state representations."
    ),
    "Cingulo-Opercular": (
        "Salience network (SN; cingulo-opercular network/CON; midcingulo-insular network) [SEP] "
        "Primary regions: anterior insula/frontal operculum, dorsal anterior cingulate/medial frontal cortex. "
        "Function: detects behaviorally relevant internal/external events (including interoceptive/affective signals), "
        "prioritizes attention and arousal, and drives performance monitoring and control adjustments (error/conflict/uncertainty)."
    ),
}

target_labels = list(text_labels.keys())
text_labels

# %%
_labels = (
    'language',
    'auditory',
    'default_mode',
    'frontoparietal_control',
    'attention',
    'visual',
    'motor',
    'cingulo_opercular'
)
_lookup = {k:v  for k, v in zip(_labels, target_labels)}

# %%
import torch.nn.functional as F

with torch.no_grad():
    text_embed = proj_head_text(specter(list(text_labels.values())))

text_embed = F.normalize(text_embed, dim=1)
networks_embed_norm = F.normalize(proj_head_image(networks_embed), dim=1)

sim = (networks_embed_norm @ text_embed.T)
pred = torch.argmax(sim, dim=1)
lookup = {v: k for k, v in zip(range(len(text_labels)), list(text_labels.keys()))}
label_inds = np.array([lookup[_lookup[i]] if i != "unknown" else -1 for i in labels])
mask = label_inds != -1

# %%
y_pred = pred[mask].detach().cpu().numpy().astype(int)
y_true = np.asarray(label_inds[mask]).astype(int)
(y_pred == y_true).mean()

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

y_pred = pred[mask].detach().cpu().numpy().astype(int)
y_true = np.asarray(label_inds[mask]).astype(int)

class_names = list(text_labels.keys())
class_names[3] = 'Frontoparietal' # shorter name
C = len(class_names)

if y_true.min() < 0 or y_true.max() >= C:
    raise ValueError(f"y_true has values outside [0, {C-1}]: min={y_true.min()}, max={y_true.max()}")
if y_pred.min() < 0 or y_pred.max() >= C:
    raise ValueError(f"y_pred has values outside [0, {C-1}]: min={y_pred.min()}, max={y_pred.max()}")

_labels = np.arange(C, dtype=int)
pretty_labels = [" ".join([j.capitalize() for j in i.split("_")]) for i in class_names]

# counts + row-normalized proportions
cm_counts = confusion_matrix(y_true, y_pred, labels=_labels)

row_sums = cm_counts.sum(axis=1, keepdims=True)
cm_props = np.divide(
    cm_counts, row_sums,
    out=np.zeros_like(cm_counts, dtype=float),
    where=row_sums != 0
)

# plot heatmap using proportions
fig, ax = plt.subplots(figsize=(10, 10))

disp = ConfusionMatrixDisplay(confusion_matrix=cm_props, display_labels=pretty_labels)
disp.plot(
    ax=ax,
    cmap="Blues",
    colorbar=False,
    xticks_rotation=90,
    include_values=False
)

# overlay ONE text block per cell: labeled count + labeled proportion
thresh = cm_props.max() * 0.5 if cm_props.size else 0.0

for i in range(C):
    for j in range(C):
        n = cm_counts[i, j]
        p = cm_props[i, j]
        txt = f"p={p:.2f}\n(n={n})"
        color = "white" if p > thresh else "black"
        ax.text(j, i, txt, ha="center", va="center", fontsize=10, color=color, linespacing=1.2)

# colorbar for proportions
im = disp.im_
cbar = ax.figure.colorbar(
    im,
    ax=ax,
    fraction=0.035,
    pad=0.02,
    shrink=0.91,
    aspect=30,
)
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_ylabel("Proportion of true-class samples (p)", size=13)

# axis labels
ax.set_title("Confusion Matrix", size=18)
ax.set_ylabel("True", size=16)
ax.set_xlabel("Predicted", size=16)

plt.tight_layout()
# plt.savefig(evaluation_output_dir / "networks_confusion.svg", dpi=300)
plt.savefig(evaluation_output_dir / "networks_confusion.svg", dpi=300)

# %%

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

y_score = (networks_embed_norm @ text_embed.T)    # (N, C) cosine similarities
y_pred  = y_score.argmax(axis=1)            # (N,) hard prediction (int class id)

y_true = np.asarray(y_true, dtype=int)
y_score = np.asarray(y_score.detach(), dtype=float)

y_pred = y_pred[mask]
y_score = y_score[mask]

C = y_score.shape[1]
Y = label_binarize(y_true, classes=np.arange(C))  # (N, C)
# Y[:, k].shape, y_score[:, k].shape ((131,), (145,))
plt.figure(figsize=(6, 6))

for k in range(C):
    # ROC undefined if class k never appears in y_true
    if Y[:, k].sum() == 0:
        continue
    fpr, tpr, _ = roc_curve(Y[:, k], y_score[:, k])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, alpha=0.85, label=f"{class_names[k]} (AUC={roc_auc:.3f})")

plt.plot([0, 1], [0, 1], "--", color="k", label="Chance (AUC=0.50)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Network Labelling: One-vs-Rest")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(evaluation_output_dir / "networks_one_v_rest.png", dpi=300)
plt.show()

# %%
# Network list
network_labels = list([i for i in text_labels.values()])

# Specter
label_embeddings = F.normalize(specter(network_labels), dim=1)
label_embeddings = F.normalize(proj_head_text(label_embeddings), dim=1)

sim = networks_embed_norm @ label_embeddings.T

inds = sim.argsort(dim=1, descending=True)[:, :2]
primary_labels = [list(text_labels.keys())[i[0]] for i in inds]
secondary_labels = [list(text_labels.keys())[i[1]] for i in inds]

# Results
df = pd.DataFrame({
    "atlas": [i[0] for i in networks],
    "atlas_label": [i[1] for i in networks],
    "true_label": [i[1] for i in label_map],
    "predicted_label_primary": primary_labels,
    "predicted_label_secondary": secondary_labels,
})

df.iloc[:20]

# %%

# %%
