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
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
from typing import Iterable, Optional

import os
SEED = 123
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
import pandas as pd
import math
import random
import re
import numpy as np

from tqdm.auto import tqdm

from nilearn.plotting import view_img

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from neurovlm.data import load_masker, data_dir
from neurovlm.models import load_model
from neurovlm.data import load_dataset, load_latent
from neurovlm.retrieval.summarization import load_huggingface_model
from neurovlm.resources.loaders import NEURO_QWEN_REPO_ID
from neurovlm.models.adapter import InterleavedDecoderAdapter, InterleavedResidualBlock, LogitCalibrator
from neurovlm.models.qformer import QFormer, NeuroQFormer
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=False)

def make_torch_generator(seed):
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g

def seed_worker(worker_id):
    worker_seed = SEED + int(worker_id)
    random.seed(worker_seed)
    np.random.seed(worker_seed)

seed_everything(SEED)

MODEL_DATA_DIR = data_dir
MODEL_DATA_DIR.mkdir(parents=True, exist_ok=True)
TEXT_SYNTH_LESS_PATH = MODEL_DATA_DIR / "text_synth_less.parquet"
ADAPTER_PATH = MODEL_DATA_DIR / "adapter_anat_full.pt"
GROUNDED_QFORMER_PATH = MODEL_DATA_DIR / "qformer_grounded_permissive_neuro_epoch-29.pt"
CANONICAL_CHECKPOINT_PREFIX = MODEL_DATA_DIR / "qformer_canonical_bottleneck"
CANONICAL_QFORMER_PATH = MODEL_DATA_DIR / "qformer_canonical_bottleneck_epoch-29.pt"
CANONICAL_BANKS_PATH = MODEL_DATA_DIR / "canonical_semantic_banks.pt"


# %% [markdown]
# <!-- ## Canonical Network Bottleneck
#
# Network-only Q-Former training with a fixed canonical semantic projection over the training network examples. -->
#

# %%
canonical_title = {
    "Visual Cortex": "Visual Cortex",
    "Auditory Cortex": "Auditory Cortex",
    "Somatosensory Cortex": "Somatosensory Cortex",
    "Motor Cortex": "Motor Cortex",
    "Premotor Cortex": "Premotor Cortex",
    "Supplementary Motor Area": "Supplementary Motor Area",
    "Oculomotor Control": "Oculomotor Control",
    "Dorsal Attention Network": "Dorsal Attention Network",
    "Ventral Attention Network": "Ventral Attention Network",
    "Frontoparietal Control": "Frontoparietal Control Network",
    "Default Mode Network": "Default Mode Network",
    "Salience Network": "Salience Network",
    "Language Network": "Language Network",
    "Speech Production": "Speech Production",
    "Speech Perception": "Speech Perception",
    "Reading Network": "Reading Network",
    "Semantic Network": "Semantic Network",
    "Phonological Processing": "Phonological Processing",
    "Hippocampal Memory": "Hippocampal Memory",
    "Working Memory": "Working Memory",
    "Response Inhibition": "Response Inhibition",
    "Reward Network": "Reward Network",
    "Emotion Network": "Emotion Network",
    "Pain Network": "Pain Network",
    "Interoceptive Cortex": "Interoceptive Cortex",
    "Autonomic Regulation": "Autonomic Regulation",
    "Theory of Mind": "Theory of Mind",
    "Action Observation": "Action Observation",
    "Empathy Network": "Empathy Network",
    "Hippocampal Formation": "Hippocampal Formation",
    "Amygdala": "Amygdala",
    "Anterior Insula": "Anterior Insula",
    "Posterior Insula": "Posterior Insula",
    "Anterior Cingulate": "Anterior Cingulate Cortex",
    "Posterior Cingulate": "Posterior Cingulate Cortex",
    "Precuneus": "Precuneus",
    "Angular Gyrus": "Angular Gyrus",
    "Supramarginal Gyrus": "Supramarginal Gyrus",
    "Inferior Frontal Gyrus": "Inferior Frontal Gyrus",
    "Dorsolateral Prefrontal Cortex": "Dorsolateral Prefrontal Cortex",
    "Ventromedial Prefrontal Cortex": "Ventromedial Prefrontal Cortex",
    "Orbitofrontal Cortex": "Orbitofrontal Cortex",
    "Basal Ganglia": "Basal Ganglia",
    "Striatum": "Striatum",
    "Thalamus": "Thalamus",
    "Cerebellum": "Cerebellum",
    "Brainstem": "Brainstem",
    "Hypothalamus": "Hypothalamus",
    "Medial Temporal Lobe": "Medial Temporal Lobe",
    "Temporal Pole": "Temporal Pole",
    "Superior Temporal Sulcus": "Superior Temporal Sulcus",
    "Parietal Cortex": "Parietal Cortex",
    "Intraparietal Sulcus": "Intraparietal Sulcus",
    "Superior Parietal Lobule": "Superior Parietal Lobule",
    "Inferior Parietal Lobule": "Inferior Parietal Lobule",
    "Motor Learning": "Motor Learning",
    "Procedural Learning": "Procedural Learning",
    "Spatial Navigation": "Spatial Navigation",
    "Retrosplenial Cortex": "Retrosplenial Cortex",
    "Parahippocampal Cortex": "Parahippocampal Cortex",
    "Olfaction": "Olfaction",
    "Gustation": "Gustation",
    "Vestibular Processing": "Vestibular Processing",
    "Music Processing": "Music Processing",
    "Numerical Cognition": "Numerical Cognition",
    "Arousal Network": "Arousal Network",
    "Limbic Network": "Limbic Network",
    "Subcortical Network": "Subcortical Network",
    "Visual Processing": "Visual Processing",
    "Primary Visual Cortex": "Primary Visual Cortex",
    "Extrastriate Visual Cortex": "Extrastriate Visual Cortex",
    "Ventral Visual Stream": "Ventral Visual Stream",
    "Dorsal Visual Stream": "Dorsal Visual Stream",
    "Object Recognition": "Object Recognition",
    "Face Processing": "Face Processing",
    "Body Perception": "Body Perception",
    "Scene Perception": "Scene Perception",
    "Motion Perception": "Motion Perception",
    "Color Perception": "Color Perception",
    "Word Form Processing": "Visual Word Form Processing",
    "Auditory Processing": "Auditory Processing",
    "Primary Auditory Cortex": "Primary Auditory Cortex",
    "Voice Perception": "Voice Perception",
    "Somatosensory Processing": "Somatosensory Processing",
    "Primary Somatosensory Cortex": "Primary Somatosensory Cortex",
    "Motor Execution": "Motor Execution",
    "Primary Motor Cortex": "Primary Motor Cortex",
    "Motor Planning": "Motor Planning",
    "Hand Movement": "Hand Movement",
    "Eye Movements": "Eye Movements",
    "Visual Attention": "Visual Attention",
    "Sustained Attention": "Sustained Attention",
    "Verbal Working Memory": "Verbal Working Memory",
    "Spatial Working Memory": "Spatial Working Memory",
    "Executive Control": "Executive Control",
    "Frontoparietal Network": "Frontoparietal Control Network",
    "Cognitive Flexibility": "Cognitive Flexibility",
    "Conflict Monitoring": "Conflict Monitoring",
    "Error Monitoring": "Error Monitoring",
    "Decision Making": "Decision Making",
    "Reward Processing": "Reward Processing",
    "Value Representation": "Value Representation",
    "Reinforcement Learning": "Reinforcement Learning",
    "Habit Learning": "Habit Learning",
    "Emotion Processing": "Emotion Processing",
    "Fear Processing": "Fear Processing",
    "Emotion Regulation": "Emotion Regulation",
    "Interoception": "Interoception",
    "Pain Processing": "Pain Processing",
    "Gustatory Processing": "Gustatory Processing",
    "Olfactory Processing": "Olfactory Processing",
    "Language Comprehension": "Language Comprehension",
    "Reading": "Reading",
    "Semantic Processing": "Semantic Processing",
    "Syntax Processing": "Syntax Processing",
    "Lexical Retrieval": "Lexical Retrieval",
    "Episodic Memory": "Episodic Memory",
    "Memory Encoding": "Memory Encoding",
    "Memory Retrieval": "Memory Retrieval",
    "Semantic Memory": "Semantic Memory",
    "Autobiographical Memory": "Autobiographical Memory",
    "Prospective Thinking": "Prospective Thinking",
    "Mental Imagery": "Mental Imagery",
    "Navigation": "Spatial Navigation",
    "Spatial Cognition": "Spatial Cognition",
    "Social Cognition": "Social Cognition",
    "Biological Motion": "Biological Motion",
    "Empathy": "Empathy",
    "Calculation": "Calculation",
    "Multisensory Integration": "Multisensory Integration",
    "Conscious Awareness": "Conscious Awareness",
    "Arousal": "Arousal",
    "Cerebellar Function": "Cerebellar Function",
    "Thalamic Function": "Thalamic Function",
    "Hippocampal Function": "Hippocampal Function",
    "Amygdala Function": "Amygdala Function",
    "Insula Function": "Insula Function",
    "Prefrontal Cortex": "Prefrontal Cortex",
    "Dorsolateral Prefrontal": "Dorsolateral Prefrontal Cortex",
    "Temporal Cortex": "Temporal Cortex",
    "Occipital Cortex": "Occipital Cortex",
    "Fusiform Gyrus": "Fusiform Gyrus",
    "Temporoparietal Junction": "Temporoparietal Junction",
    "Lateral Occipital Cortex": "Lateral Occipital Cortex",
    "Face Network": "Face Perception Network",
    "Scene Network": "Scene Perception Network",
    "Parahippocampal Place Area": "Parahippocampal Place Area",
    "Visual Word Form": "Visual Word Form Area",
    "Motion Cortex": "Motion-Selective Visual Cortex",
    "Color Cortex": "Color-Selective Visual Cortex",
    "Auditory Association Cortex": "Auditory Association Cortex",
    "Superior Temporal Gyrus": "Superior Temporal Gyrus",
    "Voice Cortex": "Voice-Selective Auditory Cortex",
    "Secondary Somatosensory Cortex": "Secondary Somatosensory Cortex",
    "Frontal Eye Fields": "Frontal Eye Fields",
    "Posterior Parietal Cortex": "Posterior Parietal Cortex",
    "Posterior Cingulate Cortex": "Posterior Cingulate Cortex",
    "Medial Prefrontal Cortex": "Medial Prefrontal Cortex",
    "Ventrolateral Prefrontal Cortex": "Ventrolateral Prefrontal Cortex",
    "Broca Area": "Broca's Area",
    "Wernicke Area": "Wernicke's Area",
    "Middle Temporal Gyrus": "Middle Temporal Gyrus",
    "Inferior Temporal Cortex": "Inferior Temporal Cortex",
    "Anterior Cingulate Cortex": "Anterior Cingulate Cortex",
    "Dorsal Anterior Cingulate": "Dorsal Anterior Cingulate Cortex",
    "Subgenual Cingulate": "Subgenual Cingulate Cortex",
    "Insular Cortex": "Insular Cortex",
    "Cingulo-Opercular Network": "Cingulo-Opercular Network",
    "Somatomotor Network": "Sensorimotor Network",
    "Visual Network": "Visual Network",
    "Auditory Network": "Auditory Network",
    "Hippocampus": "Hippocampus",
    "Entorhinal Cortex": "Entorhinal Cortex",
    "Caudate Nucleus": "Caudate Nucleus",
    "Putamen": "Putamen",
    "Ventral Striatum": "Ventral Striatum",
    "Globus Pallidus": "Globus Pallidus",
    "Mediodorsal Thalamus": "Mediodorsal Thalamus",
    "Pulvinar": "Pulvinar",
    "Midbrain": "Midbrain",
    "Pons": "Pons",
    "Medulla": "Medulla",
    "Cerebellar Vermis": "Cerebellar Vermis",
    "Cerebellar Hemispheres": "Cerebellar Hemispheres",
    "Dentate Nucleus": "Dentate Nucleus",
    "Primary Gustatory Cortex": "Primary Gustatory Cortex",
    "Olfactory Cortex": "Olfactory Cortex",
    "Vestibular Cortex": "Vestibular Cortex",
    "Memory Network": "Memory Network",
    "Speech Production Network": "Speech Production Network",
    "Speech Perception Network": "Speech Perception Network",
    "Number Network": "Numerical Cognition Network",
    "Navigation Network": "Spatial Navigation Network",
    "Body Representation": "Body Representation",
    "Hand Motor Cortex": "Hand Motor Cortex",
    "Foot Motor Cortex": "Foot Motor Cortex",
    "Face Motor Cortex": "Face Motor Cortex",
    "Paracentral Lobule": "Paracentral Lobule",
    "Frontal Operculum": "Frontal Operculum",
    "Parietal Operculum": "Parietal Operculum",
    "Cuneus": "Cuneus",
    "Lingual Gyrus": "Lingual Gyrus",
    "Precentral Gyrus": "Precentral Gyrus",
    "Postcentral Gyrus": "Postcentral Gyrus",
    "Middle Frontal Gyrus": "Middle Frontal Gyrus",
    "Superior Frontal Gyrus": "Superior Frontal Gyrus",
    "Inferior Temporal Gyrus": "Inferior Temporal Gyrus",
    "Calcarine Cortex": "Calcarine Cortex",
    "Caudate": "Caudate Nucleus",
    "Nucleus Accumbens": "Nucleus Accumbens",
    "Frontoparietal Control Network": "Frontoparietal Control Network",
    "Sensorimotor Network": "Sensorimotor Network",
    "Dorsal Anterior Cingulate Cortex": "Dorsal Anterior Cingulate Cortex",
    "Inferior Parietal Cortex": "Inferior Parietal Cortex",
    "Parahippocampal Gyrus": "Parahippocampal Gyrus",
    "Subgenual Cingulate Cortex": "Subgenual Cingulate Cortex",
    "Frontal Pole": "Frontal Pole",
    "Default Mode Network (DMN)": "Default Mode Network",
    "Default Network": "Default Mode Network",
    "DMN": "Default Mode Network",
    "FPN": "Frontoparietal Control Network",
    "Central Executive Network (CEN)": "Frontoparietal Control Network",
    "CEN": "Frontoparietal Control Network",
    "Executive Control Network (ECN)": "Frontoparietal Control Network",
    "ECN": "Frontoparietal Control Network",
    "Cognitive Control Network (CCN)": "Frontoparietal Control Network",
    "CCN": "Frontoparietal Control Network",
    "Multiple Demand Network (MDN)": "Multiple Demand Network",
    "MDN": "Multiple Demand Network",
    "Salience Network (SN)": "Salience Network",
    "SN": "Salience Network",
    "Cingulo-Opercular Network (CON)": "Cingulo-Opercular Network",
    "CON": "Cingulo-Opercular Network",
    "Salience/Ventral Attention Network": "Salience/Ventral Attention Network",
    "Ventral Attention Network (VAN)": "Ventral Attention Network",
    "Dorsal Attention Network (DAN)": "Dorsal Attention Network",
    "DAN": "Dorsal Attention Network",
    "Primary Visual Network": "Primary Visual Network",
    "Visual Association Network": "Visual Association Network",
    "Primary Auditory Network": "Primary Auditory Network",
    "Auditory Association Network": "Auditory Association Network",
    "SMN": "Sensorimotor Network",
    "Motor Network": "Sensorimotor Network",
    "Hand Sensorimotor Network": "Hand Sensorimotor Network",
    "Foot Sensorimotor Network": "Foot Sensorimotor Network",
    "Mouth Sensorimotor Network": "Mouth Sensorimotor Network",
    "Language Network (LN)": "Language Network",
    "Frontotemporal Language Network": "Language Network",
    "Dorsal Language Stream": "Dorsal Language Stream",
    "Ventral Language Stream": "Ventral Language Stream",
    "Speech Network": "Speech Network",
    "Affective Network": "Affective Network",
    "Orbitofrontal-Affective Network": "Orbitofrontal-Affective Network",
    "Medial Temporal Memory Network": "Medial Temporal Memory Network",
    "Hippocampal Network": "Hippocampal Network",
    "Posterior Medial Network": "Posterior Medial Network",
    "Scene Perception Network": "Scene Perception Network",
    "Face Perception Network": "Face Perception Network",
    "Object Perception Network": "Object Perception Network",
    "Object Network": "Object Perception Network",
    "Theory of Mind Network": "Theory of Mind Network",
    "Social Cognition Network": "Social Cognition Network",
    "Interoceptive Network": "Interoceptive Network",
    "Oculomotor Network": "Oculomotor Network",
    "Visuospatial Network": "Visuospatial Network",
    "Basal Ganglia Network": "Basal Ganglia Network",
    "Fronto-Striatal Network": "Fronto-Striatal Network",
    "Thalamic Network": "Thalamic Network",
    "Cerebellar Network": "Cerebellar Network",
    "Cerebro-Cerebellar Network": "Cerebro-Cerebellar Network",
}
concept_level = {
    "Supplementary Motor Area": "region",
    "Precuneus": "region",
    "Orbitofrontal Cortex": "region",
    "Amygdala": "region",
    "Inferior Frontal Gyrus": "region",
    "Thalamus": "region",
    "Fusiform Gyrus": "region",
    "Posterior Cingulate Cortex": "region",
    "Superior Temporal Gyrus": "region",
    "Medial Prefrontal Cortex": "region",
    "Middle Temporal Gyrus": "region",
    "Hippocampus": "region",
    "Putamen": "region",
    "Nucleus Accumbens": "region",
    "Premotor Cortex": "region",
    "Dorsal Attention Network": "network",
    "Ventral Attention Network": "network",
    "Default Mode Network": "network",
    "Language Network": "network",
    "Angular Gyrus": "region",
    "Dorsolateral Prefrontal Cortex": "region",
    "Parahippocampal Cortex": "region",
    "Salience Network": "network",
    "Anterior Insula": "region",
    "Ventromedial Prefrontal Cortex": "region",
    "Cerebellum": "region",
    "Brainstem": "region",
    "Hypothalamus": "region",
    "Intraparietal Sulcus": "region",
    "Temporoparietal Junction": "region",
    "Visual Network": "network",
    "Auditory Network": "network",
    "Striatum": "region",
    "Superior Parietal Lobule": "region",
    "Inferior Parietal Lobule": "region",
    "Lateral Occipital Cortex": "region",
    "Frontal Eye Fields": "region",
    "Posterior Parietal Cortex": "region",
    "Anterior Cingulate Cortex": "region",
    "Insular Cortex": "region",
    "Caudate Nucleus": "region",
    "Cuneus": "region",
    "Lingual Gyrus": "region",
    "Frontoparietal Control Network": "network",
    "Sensorimotor Network": "network",
    "Dorsal Anterior Cingulate Cortex": "region",
    "Subgenual Cingulate Cortex": "region",
    "Precentral Gyrus": "region",
    "Postcentral Gyrus": "region",
    "Middle Frontal Gyrus": "region",
    "Superior Frontal Gyrus": "region",
    "Inferior Temporal Gyrus": "region",
    "Calcarine Cortex": "region",
    "Caudate": "region",
    "Inferior Parietal Cortex": "region",
    "Parahippocampal Gyrus": "region",
    "Frontal Pole": "region",
    "Reading Network": "network",
    "Semantic Network": "network",
    "Reward Network": "network",
    "Emotion Network": "network",
    "Pain Network": "network",
    "Basal Ganglia": "region",
    "Superior Temporal Sulcus": "region",
    "Retrosplenial Cortex": "region",
    "Limbic Network": "network",
    "Ventral Visual Stream": "network",
    "Dorsal Visual Stream": "network",
    "Navigation Network": "network",
    "Frontoparietal Control": "network",
    "Speech Production": "function",
    "Speech Perception": "function",
    "Phonological Processing": "function",
    "Working Memory": "function",
    "Response Inhibition": "function",
    "Theory of Mind": "function",
    "Posterior Insula": "region",
    "Anterior Cingulate": "region",
    "Supramarginal Gyrus": "region",
    "Temporal Pole": "region",
    "Parietal Cortex": "region",
    "Vestibular Processing": "function",
    "Music Processing": "function",
    "Numerical Cognition": "function",
    "Arousal Network": "network",
    "Subcortical Network": "network",
    "Primary Visual Cortex": "region",
    "Extrastriate Visual Cortex": "region",
    "Primary Auditory Cortex": "region",
    "Primary Somatosensory Cortex": "region",
    "Primary Motor Cortex": "region",
    "Frontoparietal Network": "network",
    "Face Network": "network",
    "Somatomotor Network": "network",
    "Executive Control Network (ECN)": "network",
    "Salience Network (SN)": "network",
    "Cingulo-Opercular Network (CON)": "network",
    "Scene Perception Network": "network",
    "Social Cognition Network": "network",
    "Visual Cortex": "region",
    "Auditory Cortex": "region",
    "Somatosensory Cortex": "region",
    "Motor Cortex": "region",
    "Oculomotor Control": "function",
    "Hippocampal Memory": "function",
    "Interoceptive Cortex": "region",
    "Autonomic Regulation": "function",
    "Action Observation": "function",
    "Empathy Network": "network",
    "Hippocampal Formation": "region",
    "Posterior Cingulate": "region",
    "Medial Temporal Lobe": "region",
    "Motor Learning": "function",
    "Procedural Learning": "function",
    "Spatial Navigation": "function",
    "Olfaction": "function",
    "Gustation": "function",
    "Visual Processing": "function",
    "Object Recognition": "function",
    "Face Processing": "function",
    "Body Perception": "function",
    "Motion Perception": "function",
    "Color Perception": "function",
    "Word Form Processing": "function",
    "Auditory Processing": "function",
    "Voice Perception": "function",
    "Somatosensory Processing": "function",
    "Motor Execution": "function",
    "Motor Planning": "function",
    "Hand Movement": "function",
    "Eye Movements": "function",
    "Visual Attention": "function",
    "Sustained Attention": "function",
    "Verbal Working Memory": "function",
    "Spatial Working Memory": "function",
    "Cognitive Flexibility": "function",
    "Conflict Monitoring": "function",
    "Error Monitoring": "function",
    "Decision Making": "function",
    "Reward Processing": "function",
    "Value Representation": "function",
    "Reinforcement Learning": "function",
    "Habit Learning": "function",
    "Emotion Processing": "function",
    "Fear Processing": "function",
    "Emotion Regulation": "function",
    "Interoception": "function",
    "Pain Processing": "function",
    "Gustatory Processing": "function",
    "Olfactory Processing": "function",
    "Language Comprehension": "function",
    "Semantic Processing": "function",
    "Syntax Processing": "function",
    "Lexical Retrieval": "function",
    "Episodic Memory": "function",
    "Memory Encoding": "function",
    "Memory Retrieval": "function",
    "Semantic Memory": "function",
    "Autobiographical Memory": "function",
    "Prospective Thinking": "function",
    "Mental Imagery": "function",
    "Spatial Cognition": "function",
    "Biological Motion": "function",
    "Empathy": "function",
    "Calculation": "function",
    "Multisensory Integration": "function",
    "Conscious Awareness": "function",
    "Cerebellar Function": "function",
    "Thalamic Function": "function",
    "Hippocampal Function": "function",
    "Amygdala Function": "function",
    "Insula Function": "function",
    "Prefrontal Cortex": "region",
    "Temporal Cortex": "region",
    "Occipital Cortex": "region",
    "Scene Network": "network",
    "Parahippocampal Place Area": "region",
    "Visual Word Form": "region",
    "Motion Cortex": "region",
    "Color Cortex": "region",
    "Auditory Association Cortex": "region",
    "Voice Cortex": "region",
    "Secondary Somatosensory Cortex": "region",
    "Ventrolateral Prefrontal Cortex": "region",
    "Broca Area": "region",
    "Wernicke Area": "region",
    "Inferior Temporal Cortex": "region",
    "Entorhinal Cortex": "region",
    "Ventral Striatum": "region",
    "Globus Pallidus": "region",
    "Mediodorsal Thalamus": "region",
    "Pulvinar": "region",
    "Midbrain": "region",
    "Pons": "region",
    "Medulla": "region",
    "Cerebellar Vermis": "region",
    "Cerebellar Hemispheres": "region",
    "Dentate Nucleus": "region",
    "Primary Gustatory Cortex": "region",
    "Olfactory Cortex": "region",
    "Vestibular Cortex": "region",
    "Memory Network": "network",
    "Speech Production Network": "network",
    "Speech Perception Network": "network",
    "Number Network": "network",
    "Body Representation": "function",
    "Hand Motor Cortex": "region",
    "Foot Motor Cortex": "region",
    "Face Motor Cortex": "region",
    "Paracentral Lobule": "region",
    "Frontal Operculum": "region",
    "Parietal Operculum": "region",
    "Default Mode Network (DMN)": "network",
    "Default Network": "network",
    "DMN": "network",
    "FPN": "network",
    "Central Executive Network (CEN)": "network",
    "CEN": "network",
    "ECN": "network",
    "Cognitive Control Network (CCN)": "network",
    "CCN": "network",
    "Multiple Demand Network (MDN)": "network",
    "MDN": "network",
    "SN": "network",
    "CON": "network",
    "Salience/Ventral Attention Network": "network",
    "Ventral Attention Network (VAN)": "network",
    "Dorsal Attention Network (DAN)": "network",
    "DAN": "network",
    "Primary Visual Network": "network",
    "Visual Association Network": "network",
    "Primary Auditory Network": "network",
    "Auditory Association Network": "network",
    "SMN": "network",
    "Motor Network": "network",
    "Hand Sensorimotor Network": "network",
    "Foot Sensorimotor Network": "network",
    "Mouth Sensorimotor Network": "network",
    "Language Network (LN)": "network",
    "Frontotemporal Language Network": "network",
    "Dorsal Language Stream": "network",
    "Ventral Language Stream": "network",
    "Speech Network": "network",
    "Affective Network": "network",
    "Orbitofrontal-Affective Network": "network",
    "Medial Temporal Memory Network": "network",
    "Hippocampal Network": "network",
    "Posterior Medial Network": "network",
    "Face Perception Network": "network",
    "Object Perception Network": "network",
    "Object Network": "network",
    "Theory of Mind Network": "network",
    "Interoceptive Network": "network",
    "Oculomotor Network": "network",
    "Visuospatial Network": "network",
    "Basal Ganglia Network": "network",
    "Fronto-Striatal Network": "network",
    "Thalamic Network": "network",
    "Cerebellar Network": "network",
    "Cerebro-Cerebellar Network": "network",
    "Scene Perception": "function",
    "Executive Control": "function",
    "Reading": "function",
    "Navigation": "function",
    "Social Cognition": "function",
    "Arousal": "function",
    "Dorsolateral Prefrontal": "region",
    "Dorsal Anterior Cingulate": "region",
    "Subgenual Cingulate": "region",
    "Cingulo-Opercular Network": "network",
}
function_desc = {
  "Visual Cortex": "Visual Perception\nVisual perception is the interpretation of visual input into organized information about form, color, motion, and spatial layout. It primarily involves primary and extrastriate visual cortex in the occipital lobe, with extensions into ventral temporal and dorsal parietal visual pathways. This concept is relevant for recognizing objects, guiding attention, and constructing coherent representations of the visual environment.",
  "Auditory Cortex": "Auditory Perception\nAuditory perception is the processing of sound features such as pitch, rhythm, intensity, and temporal structure. It involves primary auditory cortex in Heschl's gyrus, surrounding superior temporal cortex, and auditory association regions along the lateral temporal lobe. This concept supports speech perception, environmental sound recognition, and the organization of sound into meaningful auditory events.",
  "Somatosensory Cortex": "Somatosensory Perception\nSomatosensory perception is the representation of touch, pressure, vibration, temperature, and body position. It involves primary somatosensory cortex in the postcentral gyrus, secondary somatosensory cortex, posterior parietal cortex, and insular regions. This concept supports tactile recognition, body awareness, and the integration of sensory feedback for action.",
  "Motor Cortex": "Voluntary Movement\nVoluntary movement is the planning and execution of intentional body actions. It involves primary motor cortex in the precentral gyrus, premotor cortex, supplementary motor area, basal ganglia, cerebellum, and thalamus. This concept is relevant for controlling force, timing, coordination, and the selection of goal-directed movements.",
  "Premotor Cortex": "Action Planning\nAction planning is the preparation of movements based on goals, sensory cues, and context. It involves premotor cortex, supplementary motor area, posterior parietal cortex, basal ganglia, and primary motor cortex. This concept supports selecting actions, mapping stimuli to responses, and organizing movements before execution.",
  "Supplementary Motor Area": "Motor Sequencing\nMotor sequencing is the organization of actions into ordered patterns across time. It involves the supplementary motor area on the medial frontal cortex, pre-supplementary motor area, basal ganglia, primary motor cortex, and cerebellum. This concept supports internally guided movement, sequential action, and coordination of complex motor routines.",
  "Dorsal Attention Network": "Goal-Directed Attention\nGoal-directed attention is the voluntary selection of information based on current goals. It involves the frontal eye fields, intraparietal sulcus, superior parietal lobule, and dorsal posterior parietal cortex. This concept supports spatial orienting, visual search, and maintaining attention toward behaviorally relevant locations or features.",
  "Ventral Attention Network": "Attention Reorienting\nAttention reorienting is the detection of unexpected or behaviorally relevant events that require a shift in focus. It involves the temporoparietal junction, ventral frontal cortex, inferior frontal gyrus, frontal operculum, and anterior insula. This concept supports interrupting ongoing attention when salient external information needs to be processed.",
  "Frontoparietal Control Network": "Cognitive Control\nCognitive control is the flexible regulation of thought and behavior according to current goals. It involves dorsolateral prefrontal cortex, inferior parietal lobule, intraparietal sulcus, anterior prefrontal cortex, and lateral frontal regions. This concept supports task switching, working memory, decision control, and adaptive coordination across cognitive systems.",
  "Default Mode Network": "Internally Directed Cognition\nInternally directed cognition is the generation of thoughts that are not tied directly to immediate sensory demands. It involves medial prefrontal cortex, posterior cingulate cortex, precuneus, angular gyrus, lateral temporal cortex, and medial temporal lobe regions. This concept supports autobiographical memory, future thinking, self-referential thought, and broad social inference.",
  "Salience Network": "Salience Detection\nSalience detection is the identification of information that is behaviorally important or internally significant. It involves the anterior insula, dorsal anterior cingulate cortex, frontal operculum, amygdala, thalamus, and striatum. This concept supports prioritizing attention, coordinating control systems, and linking bodily signals with action-relevant decisions.",
  "Language Network": "Language Processing\nLanguage processing is the comprehension and production of meaningful spoken, written, or symbolic communication. It involves inferior frontal gyrus, superior temporal gyrus, middle temporal gyrus, angular gyrus, supramarginal gyrus, and temporal-parietal language regions. This concept supports word meaning, sentence structure, speech comprehension, and expressive communication.",
  "Reading Network": "Reading\nReading is the transformation of written symbols into sounds, meanings, and linguistic structure. It involves the visual word form area, occipitotemporal cortex, inferior frontal gyrus, temporoparietal cortex, angular gyrus, and superior temporal regions. This concept supports fluent word recognition, phonological decoding, and access to semantic knowledge from text.",
  "Semantic Network": "Semantic Cognition\nSemantic cognition is the representation and retrieval of conceptual meaning across words, objects, actions, and experiences. It involves anterior temporal cortex, middle temporal gyrus, angular gyrus, inferior frontal gyrus, medial temporal regions, and posterior association cortex. This concept supports understanding concepts, linking related knowledge, and using meaning to guide thought and communication.",
  "Reward Network": "Reward Processing\nReward processing is the evaluation of motivational value, reinforcement, and expected outcomes. It involves the ventral striatum, nucleus accumbens, orbitofrontal cortex, ventromedial prefrontal cortex, amygdala, midbrain, and anterior cingulate cortex. This concept supports learning from outcomes, selecting valued actions, and updating behavior based on incentives.",
  "Emotion Network": "Emotional Processing\nEmotional processing is the interpretation and regulation of affective significance in internal and external events. It involves the amygdala, anterior insula, anterior cingulate cortex, orbitofrontal cortex, ventromedial prefrontal cortex, hypothalamus, and brainstem. This concept supports evaluating valence, shaping bodily responses, and guiding behavior in emotionally meaningful contexts.",
  "Pain Network": "Pain Perception\nPain perception is the processing of nociceptive and affective signals related to bodily threat or discomfort. It involves the anterior insula, posterior insula, anterior cingulate cortex, thalamus, somatosensory cortex, prefrontal cortex, and brainstem. This concept supports sensory localization of pain, appraisal of unpleasantness, and adaptive protective behavior.",
  "Interoceptive Cortex": "Interoception\nInteroception is the perception and interpretation of signals arising from inside the body. It involves the posterior insula, anterior insula, anterior cingulate cortex, somatosensory cortex, hypothalamus, and brainstem autonomic regions. This concept supports bodily awareness, affective feeling states, and regulation of internal physiological needs.",
  "Empathy Network": "Empathy\nEmpathy is the capacity to represent and respond to the internal states of others. It involves anterior insula, anterior cingulate cortex, medial prefrontal cortex, temporoparietal junction, superior temporal sulcus, and amygdala. This concept supports affect sharing, perspective taking, and socially appropriate responses to other people.",
  "Hippocampal Formation": "Episodic Memory\nEpisodic memory is the encoding and retrieval of events situated in specific contexts. It involves the hippocampus, dentate gyrus, subiculum, entorhinal cortex, parahippocampal cortex, and retrosplenial cortex. This concept supports remembering experiences, binding items to places, and constructing event-based mental representations.",
  "Amygdala": "Affective Salience\nAffective salience is the evaluation of stimuli according to emotional and motivational significance. It involves the amygdala, anterior insula, orbitofrontal cortex, ventromedial prefrontal cortex, hippocampus, hypothalamus, and brainstem. This concept supports rapid appraisal of relevance, emotional learning, and modulation of attention and memory.",
  "Anterior Insula": "Subjective Awareness\nSubjective awareness is the conscious integration of bodily, emotional, and attentional signals. It involves the anterior insula, frontal operculum, dorsal anterior cingulate cortex, orbitofrontal cortex, and somatosensory-related regions. This concept supports awareness of internal states, salience detection, and flexible shifts between cognitive and affective demands.",
  "Posterior Insula": "Bodily Sensation\nBodily sensation is the representation of physical signals from the body surface and internal organs. It involves the posterior insula, secondary somatosensory cortex, thalamus, parietal operculum, and primary somatosensory cortex. This concept supports tactile awareness, visceral sensation, pain localization, and the sensory foundation of interoception.",
  "Anterior Cingulate Cortex": "Performance Monitoring\nPerformance monitoring is the evaluation of actions, conflict, effort, and outcomes during behavior. It involves anterior cingulate cortex, medial prefrontal cortex, anterior insula, dorsolateral prefrontal cortex, basal ganglia, and thalamus. This concept supports detecting the need for control, adjusting behavior, and coordinating attention with action demands.",
  "Posterior Cingulate Cortex": "Self-Related Memory\nSelf-related memory is the integration of personal relevance with remembered and imagined experience. It involves posterior cingulate cortex, precuneus, medial prefrontal cortex, angular gyrus, retrosplenial cortex, and medial temporal lobe. This concept supports autobiographical thought, scene construction, and internally oriented cognition.",
  "Precuneus": "Mental Imagery\nMental imagery is the construction of internal representations of scenes, perspectives, and body-centered information. It involves the precuneus, posterior cingulate cortex, superior parietal lobule, medial prefrontal cortex, and occipital-parietal association regions. This concept supports visuospatial imagery, self-related perspective, and internally generated simulations.",
  "Angular Gyrus": "Conceptual Integration\nConceptual integration is the combination of semantic, spatial, social, and episodic information into coherent meaning. It involves the angular gyrus, inferior parietal lobule, posterior temporal cortex, medial temporal lobe, and default mode regions. This concept supports comprehension, memory retrieval, number meaning, and flexible association across domains.",
  "Supramarginal Gyrus": "Phonological and Sensorimotor Integration\nPhonological and sensorimotor integration is the linking of sound-based language representations with action and body-related information. It involves the supramarginal gyrus, inferior parietal lobule, superior temporal cortex, inferior frontal gyrus, and premotor regions. This concept supports speech sound processing, verbal working memory, imitation, and action-related aspects of communication.",
  "Inferior Frontal Gyrus": "Controlled Language Selection\nControlled language selection is the regulation of speech, meaning, and response choices during communication. It involves inferior frontal gyrus, especially pars opercularis and pars triangularis, along with temporal language cortex, premotor regions, and dorsolateral prefrontal cortex. This concept supports word retrieval, syntactic processing, inhibition of competing responses, and expressive language control.",
  "Dorsolateral Prefrontal Cortex": "Executive Control\nExecutive control is the active maintenance and manipulation of information to guide behavior. It involves dorsolateral prefrontal cortex, frontoparietal control regions, anterior cingulate cortex, basal ganglia, and posterior parietal cortex. This concept supports working memory, planning, rule use, and flexible adjustment to task demands.",
  "Ventromedial Prefrontal Cortex": "Value-Based Judgment\nValue-based judgment is the integration of affective, social, and reward information into preferences and choices. It involves ventromedial prefrontal cortex, orbitofrontal cortex, amygdala, ventral striatum, posterior cingulate cortex, and medial temporal regions. This concept supports evaluating personal relevance, estimating subjective value, and guiding decisions in meaningful contexts.",
  "Orbitofrontal Cortex": "Outcome Evaluation\nOutcome evaluation is the representation of reward, punishment, preference, and changing value. It involves orbitofrontal cortex, ventromedial prefrontal cortex, amygdala, ventral striatum, insula, and sensory association cortices. This concept supports adaptive choice, updating expectations, and linking sensory information with motivational significance.",
  "Basal Ganglia": "Action Selection\nAction selection is the process of choosing and initiating appropriate actions while suppressing alternatives. It involves the striatum, globus pallidus, subthalamic nucleus, substantia nigra, thalamus, and frontal cortex. This concept supports habit learning, motor control, reward-guided behavior, and sequencing of cognitive and motor operations.",
  "Striatum": "Reinforcement Learning\nReinforcement learning is the adjustment of behavior based on rewards, feedback, and prediction errors. It involves the caudate nucleus, putamen, ventral striatum, nucleus accumbens, dopaminergic midbrain, and prefrontal cortex. This concept supports learning action values, forming habits, and selecting behaviors that match expected outcomes.",
  "Thalamus": "Thalamocortical Integration\nThalamocortical integration is the coordination of information flow between subcortical systems and the cerebral cortex. It involves thalamic nuclei such as mediodorsal thalamus, pulvinar, ventral posterior nuclei, lateral geniculate nucleus, and their cortical projection targets. This concept supports attention, sensory relay, arousal regulation, and flexible communication across cortical networks.",
  "Cerebellum": "Coordination and Prediction\nCoordination and prediction involve the calibration of timing, error correction, and smooth adjustment of behavior. They involve the cerebellar hemispheres, cerebellar vermis, dentate nucleus, deep cerebellar nuclei, brainstem, thalamus, and frontal-parietal cortical regions. This concept supports motor coordination, sequence learning, cognitive timing, and predictive control.",
  "Brainstem": "Arousal and Autonomic Regulation\nArousal and autonomic regulation maintain wakefulness, basic bodily functions, and readiness for action. They involve the midbrain, pons, medulla, reticular formation, locus coeruleus, raphe nuclei, parabrachial regions, and connections with hypothalamus and thalamus. This concept supports alertness, respiration, cardiovascular regulation, and global modulation of brain state.",
  "Hypothalamus": "Homeostatic Regulation\nHomeostatic regulation is the control of internal bodily states needed for survival and stability. It involves the hypothalamus, pituitary connections, brainstem autonomic centers, amygdala, insula, and limbic forebrain regions. This concept supports hunger, thirst, temperature regulation, endocrine signaling, stress responses, and motivated behavior.",
  "Medial Temporal Lobe": "Declarative Memory\nDeclarative memory is the formation and retrieval of facts and experiences that can be consciously accessed. It involves the hippocampus, entorhinal cortex, perirhinal cortex, parahippocampal cortex, and broader medial temporal lobe structures. This concept supports learning new information, recognizing familiar items, and binding experiences into coherent memories.",
  "Temporal Pole": "Social Semantic Knowledge\nSocial semantic knowledge is the representation of meaning about people, emotions, concepts, and socially relevant categories. It involves the temporal pole, anterior temporal cortex, amygdala, orbitofrontal cortex, medial prefrontal cortex, and superior temporal regions. This concept supports person knowledge, conceptual associations, and interpretation of socially meaningful information.",
  "Superior Temporal Sulcus": "Biological Motion Perception\nBiological motion perception is the interpretation of socially relevant movement such as gaze, facial motion, and body actions. It involves the superior temporal sulcus, temporoparietal junction, fusiform gyrus, inferior frontal regions, and posterior temporal cortex. This concept supports understanding observed actions, social signals, and dynamic cues from other people.",
  "Parietal Cortex": "Spatial Cognition\nSpatial cognition is the representation of locations, body position, quantities, and action-relevant spatial relationships. It involves posterior parietal cortex, intraparietal sulcus, superior parietal lobule, inferior parietal lobule, and connections with frontal and visual regions. This concept supports attention, reaching, numerical reasoning, visuospatial working memory, and sensorimotor integration.",
  "Intraparietal Sulcus": "Spatial Attention\nSpatial attention is the selection and prioritization of locations, features, and quantities in perceptual space. It involves the intraparietal sulcus, superior parietal lobule, frontal eye fields, posterior parietal cortex, and dorsal visual stream. This concept supports visual search, eye movement planning, numerical magnitude, and goal-directed orienting.",
  "Superior Parietal Lobule": "Visuomotor Integration\nVisuomotor integration is the transformation of visual information into body-centered coordinates for action. It involves the superior parietal lobule, intraparietal sulcus, dorsal premotor cortex, frontal eye fields, and dorsal visual stream. This concept supports reaching, spatial attention, eye-hand coordination, and online adjustment of movements.",
  "Inferior Parietal Lobule": "Multimodal Association\nMultimodal association is the integration of language, attention, memory, and spatial information. It involves the inferior parietal lobule, angular gyrus, supramarginal gyrus, posterior temporal cortex, and lateral prefrontal regions. This concept supports comprehension, attentional reorienting, body representation, and flexible conceptual processing.",
  "Retrosplenial Cortex": "Contextual Scene Memory\nContextual scene memory is the linking of places, spatial context, and remembered experience. It involves retrosplenial cortex, posterior cingulate cortex, parahippocampal cortex, hippocampus, and medial parietal regions. This concept supports navigation, landmark recognition, scene construction, and retrieval of spatial context.",
  "Parahippocampal Cortex": "Scene Context Processing\nScene context processing is the representation of places, environmental layouts, and contextual associations. It involves parahippocampal cortex, parahippocampal place area, retrosplenial cortex, hippocampus, and occipital scene-selective regions. This concept supports recognizing environments, encoding spatial context, and linking scenes with memory.",
  "Arousal Network": "Alertness\nAlertness is the regulation of wakeful readiness and responsiveness to incoming information. It involves brainstem reticular formation, locus coeruleus, basal forebrain, thalamus, hypothalamus, anterior cingulate cortex, and widespread cortical projections. This concept supports vigilance, sustained attention, and global readiness for perception and action.",
  "Limbic Network": "Affective-Mnemonic Integration\nAffective-mnemonic integration is the linking of emotion, motivation, memory, and bodily state. It involves amygdala, hippocampus, ventromedial prefrontal cortex, orbitofrontal cortex, anterior cingulate cortex, hypothalamus, and nucleus accumbens. This concept supports emotionally meaningful learning, motivated behavior, and the integration of feeling with memory.",
  "Subcortical Network": "Subcortical Regulation\nSubcortical regulation is the coordination of motor, motivational, sensory, and arousal processes below the cortex. It involves basal ganglia, thalamus, brainstem, hypothalamus, amygdala, hippocampus, and cerebellar connections. This concept supports action selection, arousal control, reinforcement learning, and routing of information to cortical systems.",
  "Primary Visual Cortex": "Early Visual Encoding\nEarly visual encoding is the initial cortical representation of edges, contrast, orientation, and retinotopic location. It involves primary visual cortex along the calcarine sulcus, lateral geniculate inputs, and nearby occipital visual areas. This concept supports precise visual mapping, low-level feature extraction, and the foundation for later object and scene perception.",
  "Extrastriate Visual Cortex": "Visual Feature Processing\nVisual feature processing is the analysis of complex visual properties beyond primary visual input. It involves extrastriate occipital cortex, lateral occipital cortex, fusiform regions, motion-sensitive cortex, color-sensitive regions, and dorsal visual areas. This concept supports recognition of objects, motion, color, shape, and spatially organized visual patterns.",
  "Ventral Visual Stream": "Object Recognition\nObject recognition is the identification of visual forms, categories, faces, and written symbols. It involves ventral occipitotemporal cortex, lateral occipital cortex, fusiform gyrus, inferior temporal cortex, and anterior temporal regions. This concept supports knowing what something is, linking visual form to meaning, and recognizing stable object identities.",
  "Dorsal Visual Stream": "Visuospatial Action\nVisuospatial action is the use of visual information to guide movement and spatial attention. It involves occipitoparietal cortex, intraparietal sulcus, superior parietal lobule, frontal eye fields, and premotor regions. This concept supports locating objects, reaching, tracking motion, and transforming visual input into action plans.",
  "Primary Auditory Cortex": "Early Auditory Encoding\nEarly auditory encoding is the initial cortical analysis of frequency, intensity, and temporal sound structure. It involves primary auditory cortex in Heschl's gyrus, medial geniculate thalamic inputs, and surrounding superior temporal regions. This concept supports basic sound perception, pitch organization, and the foundation for speech and music processing.",
  "Primary Somatosensory Cortex": "Tactile Encoding\nTactile encoding is the cortical representation of touch, pressure, vibration, and body location. It involves primary somatosensory cortex in the postcentral gyrus, thalamic somatosensory nuclei, and adjacent parietal sensory areas. This concept supports fine touch discrimination, body mapping, and sensory feedback for movement.",
  "Primary Motor Cortex": "Motor Execution\nMotor execution is the direct cortical control of voluntary muscle activity. It involves primary motor cortex in the precentral gyrus, corticospinal pathways, premotor cortex, supplementary motor area, basal ganglia, cerebellum, and spinal motor circuits. This concept supports precise movement output, force control, and somatotopic organization of body actions.",
  "Prefrontal Cortex": "Executive Function\nExecutive function is the regulation of goals, plans, decisions, and flexible behavior. It involves dorsolateral, ventrolateral, orbitofrontal, ventromedial, medial, and anterior prefrontal cortices with connections to parietal cortex, basal ganglia, and limbic regions. This concept supports working memory, inhibition, planning, value-guided choice, and adaptive control.",
  "Temporal Cortex": "Auditory and Semantic Processing\nAuditory and semantic processing links sound, language, object knowledge, and memory. It involves superior temporal gyrus, middle temporal gyrus, inferior temporal cortex, temporal pole, fusiform gyrus, and medial temporal lobe. This concept supports speech comprehension, conceptual knowledge, object recognition, and memory-related association.",
  "Occipital Cortex": "Visual Analysis\nVisual analysis is the cortical processing of visual features and spatial structure. It involves primary visual cortex, extrastriate occipital cortex, lateral occipital cortex, cuneus, lingual gyrus, and occipitotemporal visual regions. This concept supports seeing shape, color, motion, location, and visual patterns in the environment.",
  "Fusiform Gyrus": "High-Level Visual Recognition\nHigh-level visual recognition is the identification of complex visual categories such as faces, words, and objects. It involves the fusiform gyrus, ventral occipitotemporal cortex, inferior temporal cortex, lateral occipital cortex, and anterior temporal regions. This concept supports expertise-based visual recognition, category learning, and linking visual forms to meaning.",
  "Temporoparietal Junction": "Social Perspective and Reorienting\nSocial perspective and reorienting involve shifting attention toward relevant events and representing other minds. They involve the temporoparietal junction, posterior superior temporal sulcus, inferior parietal lobule, ventral frontal cortex, and medial prefrontal cortex. This concept supports attention reorienting, perspective taking, and inference about others' beliefs or intentions.",
  "Lateral Occipital Cortex": "Object Shape Perception\nObject shape perception is the analysis of visual form needed to identify objects. It involves lateral occipital cortex, ventral occipitotemporal cortex, fusiform gyrus, inferior temporal cortex, and early visual cortex. This concept supports recognizing object boundaries, matching shapes to categories, and distinguishing objects from background scenes.",
  "Face Perception Network": "Face Perception\nFace perception is the recognition and interpretation of facial identity, expression, and social cues. It involves fusiform face-selective regions, occipital face areas, superior temporal sulcus, amygdala, anterior temporal cortex, and orbitofrontal regions. This concept supports recognizing individuals, reading facial expressions, and extracting socially meaningful information from faces.",
  "Scene Perception Network": "Scene Perception\nScene perception is the representation of environmental layout, landmarks, and spatial context. It involves parahippocampal place area, retrosplenial cortex, occipital place area, hippocampus, parahippocampal cortex, and medial parietal regions. This concept supports recognizing places, navigating environments, and constructing spatial context for memory.",
  "Parahippocampal Place Area": "Place Recognition\nPlace recognition is the identification of scenes and environmental layouts. It involves the parahippocampal place area, parahippocampal cortex, retrosplenial cortex, occipital place area, and hippocampal formation. This concept supports recognizing familiar environments, encoding spatial context, and linking visual scenes with navigational memory.",
  "Visual Word Form Area": "Orthographic Recognition\nOrthographic recognition is the identification of written word forms and letter patterns. It involves the visual word form area in left ventral occipitotemporal cortex, fusiform gyrus, occipital visual cortex, temporal language regions, and inferior frontal language regions. This concept supports rapid reading, word-form recognition, and access from print to sound and meaning.",
  "Motion-Selective Visual Cortex": "Motion Perception\nMotion perception is the detection and interpretation of movement direction, speed, and dynamic visual change. It involves motion-selective visual cortex in lateral occipitotemporal regions, especially area MT or V5, with inputs from early visual cortex and dorsal parietal areas. This concept supports tracking moving objects, perceiving biological motion, and guiding visually directed action.",
  "Color-Selective Visual Cortex": "Color Perception\nColor perception is the representation of chromatic information as meaningful visual qualities. It involves color-sensitive ventral occipital and occipitotemporal regions, early visual cortex, lingual gyrus, fusiform regions, and inferior temporal cortex. This concept supports object identification, scene interpretation, and stable perception of surface properties.",
  "Auditory Association Cortex": "Complex Sound Processing\nComplex sound processing is the interpretation of meaningful auditory patterns beyond basic acoustic features. It involves auditory association cortex along superior temporal gyrus, superior temporal sulcus, planum temporale, middle temporal regions, and inferior frontal language areas. This concept supports speech comprehension, music perception, voice recognition, and environmental sound interpretation.",
  "Superior Temporal Gyrus": "Speech and Sound Processing\nSpeech and sound processing is the analysis of auditory information for linguistic and nonlinguistic meaning. It involves superior temporal gyrus, Heschl's gyrus, planum temporale, superior temporal sulcus, middle temporal gyrus, and inferior frontal language regions. This concept supports speech perception, phonological processing, and recognition of complex sounds.",
  "Voice-Selective Auditory Cortex": "Voice Perception\nVoice perception is the recognition of vocal identity, affect, and communicative cues from human sounds. It involves voice-sensitive regions of superior temporal cortex, superior temporal sulcus, auditory association cortex, amygdala, and inferior frontal regions. This concept supports recognizing speakers, interpreting vocal emotion, and extracting social meaning from speech sounds.",
  "Secondary Somatosensory Cortex": "Tactile Integration\nTactile integration is the combination of body-based sensory signals into meaningful somatosensory representations. It involves secondary somatosensory cortex in the parietal operculum, primary somatosensory cortex, posterior insula, thalamus, and posterior parietal cortex. This concept supports texture perception, bilateral tactile processing, object manipulation, and body-centered awareness.",
  "Frontal Eye Fields": "Eye Movement Control\nEye movement control is the selection and initiation of gaze shifts toward relevant locations. It involves frontal eye fields, intraparietal sulcus, superior colliculus, visual cortex, supplementary eye fields, and basal ganglia. This concept supports visual search, spatial attention, saccade planning, and coordination between attention and gaze.",
  "Posterior Parietal Cortex": "Spatial Sensorimotor Mapping\nSpatial sensorimotor mapping is the transformation of sensory information into coordinates for attention and action. It involves posterior parietal cortex, superior parietal lobule, intraparietal sulcus, inferior parietal lobule, frontal eye fields, and premotor cortex. This concept supports reaching, orienting, spatial working memory, and visually guided behavior.",
  "Medial Prefrontal Cortex": "Self-Referential Cognition\nSelf-referential cognition is the evaluation of information in relation to one's own traits, goals, and social context. It involves medial prefrontal cortex, posterior cingulate cortex, precuneus, temporoparietal junction, temporal pole, and ventromedial prefrontal cortex. This concept supports thinking about the self, interpreting social information, and assigning personal relevance.",
  "Ventrolateral Prefrontal Cortex": "Controlled Retrieval\nControlled retrieval is the strategic selection of relevant information from memory or language. It involves ventrolateral prefrontal cortex, inferior frontal gyrus, temporal cortex, parietal association regions, and dorsolateral prefrontal cortex. This concept supports semantic selection, response inhibition, verbal working memory, and flexible access to stored knowledge.",
  "Broca's Area": "Speech Production\nSpeech production is the planning and organization of spoken language output. It involves Broca's area in the left inferior frontal gyrus, premotor cortex, supplementary motor area, insula, superior temporal regions, and motor speech pathways. This concept supports articulation planning, syntactic organization, phonological sequencing, and controlled verbal expression.",
  "Wernicke's Area": "Speech Comprehension\nSpeech comprehension is the extraction of linguistic meaning from spoken language. It involves posterior superior temporal gyrus, posterior superior temporal sulcus, middle temporal gyrus, angular gyrus, and connections with inferior frontal language regions. This concept supports understanding words and sentences, mapping sounds to meaning, and integrating language context.",
  "Middle Temporal Gyrus": "Lexical-Semantic Processing\nLexical-semantic processing is the representation and retrieval of word meanings and conceptual associations. It involves middle temporal gyrus, superior temporal sulcus, inferior temporal cortex, angular gyrus, temporal pole, and inferior frontal gyrus. This concept supports vocabulary comprehension, semantic association, and integration of meaning across language and perception.",
  "Inferior Temporal Cortex": "Visual Object Knowledge\nVisual object knowledge is the linking of object appearance with category and meaning. It involves inferior temporal cortex, fusiform gyrus, lateral occipital cortex, ventral visual stream, and anterior temporal regions. This concept supports recognizing objects, distinguishing visual categories, and connecting perceptual features with conceptual knowledge.",
  "Dorsal Anterior Cingulate Cortex": "Conflict Monitoring\nConflict monitoring is the detection of competing responses, effort demands, and the need for control. It involves dorsal anterior cingulate cortex, anterior insula, dorsolateral prefrontal cortex, supplementary motor area, basal ganglia, and thalamus. This concept supports adaptive control, performance adjustment, and allocation of effort during demanding tasks.",
  "Subgenual Cingulate Cortex": "Affective Valuation\nAffective valuation is the integration of mood, bodily state, and personal value into emotional meaning. It involves subgenual cingulate cortex, ventromedial prefrontal cortex, amygdala, hypothalamus, nucleus accumbens, and brainstem autonomic regions. This concept supports emotional appraisal, regulation of affective state, and linking internal feeling with motivational priorities.",
  "Insular Cortex": "Interoceptive Awareness\nInteroceptive awareness is the integration of bodily sensation with emotion, salience, and conscious experience. It involves posterior insula, mid-insula, anterior insula, frontal operculum, anterior cingulate cortex, and somatosensory-related regions. This concept supports awareness of internal body states, affective feeling, pain processing, and salience-guided behavior.",
  "Cingulo-Opercular Network": "Sustained Task Control\nSustained task control is the maintenance of stable task goals and performance across time. It involves dorsal anterior cingulate cortex, anterior insula, frontal operculum, anterior prefrontal cortex, and thalamic regions. This concept supports vigilance, error monitoring, maintenance of task set, and stable engagement during demanding cognition.",
  "Sensorimotor Network": "Sensorimotor Control\nSensorimotor control is the integration of sensory feedback with movement execution. It involves primary motor cortex, primary somatosensory cortex, premotor cortex, supplementary motor area, paracentral lobule, basal ganglia, cerebellum, and thalamus. This concept supports voluntary movement, body representation, action timing, and coordination of sensory feedback with motor output.",
  "Visual Network": "Visual Processing\nVisual processing is the organization of visual input into features, objects, motion, and spatial structure. It involves primary visual cortex, extrastriate visual cortex, lateral occipital cortex, fusiform gyrus, dorsal visual stream, and ventral visual stream. This concept supports perception of the external environment, visual recognition, attention, and visually guided behavior.",
  "Auditory Network": "Auditory Processing\nAuditory processing is the analysis of sound structure and meaning. It involves primary auditory cortex, superior temporal gyrus, auditory association cortex, planum temporale, superior temporal sulcus, and inferior frontal auditory-language regions. This concept supports speech perception, music processing, voice recognition, and awareness of environmental sounds.",
  "Hippocampus": "Relational Memory\nRelational memory is the binding of items, places, contexts, and temporal information into coherent experiences. It involves the hippocampus, entorhinal cortex, parahippocampal cortex, retrosplenial cortex, medial prefrontal cortex, and posterior cingulate cortex. This concept supports episodic memory, spatial navigation, and flexible recombination of past experiences.",
  "Entorhinal Cortex": "Memory and Spatial Mapping\nMemory and spatial mapping link sensory information with hippocampal representations of context and location. They involve entorhinal cortex, hippocampus, parahippocampal cortex, perirhinal cortex, retrosplenial cortex, and medial temporal lobe circuits. This concept supports navigation, contextual memory, and the transformation of experience into relational memory codes.",
  "Caudate Nucleus": "Goal-Directed Action\nGoal-directed action is the selection of behavior based on rules, feedback, and expected outcomes. It involves the caudate nucleus, dorsolateral prefrontal cortex, anterior cingulate cortex, thalamus, putamen, and dopaminergic midbrain. This concept supports learning action-outcome associations, cognitive flexibility, and controlled response selection.",
  "Putamen": "Motor Habit Learning\nMotor habit learning is the acquisition of repeated action patterns that become efficient with practice. It involves the putamen, primary motor cortex, premotor cortex, supplementary motor area, globus pallidus, thalamus, and cerebellum. This concept supports procedural learning, movement sequencing, and automatic aspects of skilled behavior.",
  "Ventral Striatum": "Motivational Value\nMotivational value is the representation of reward expectation and incentive significance. It involves the ventral striatum, nucleus accumbens, orbitofrontal cortex, ventromedial prefrontal cortex, amygdala, hippocampus, and midbrain dopamine regions. This concept supports approach behavior, reward learning, and prioritization of actions with expected benefit.",
  "Globus Pallidus": "Motor Gating\nMotor gating is the regulation of movement initiation and suppression through basal ganglia output pathways. It involves the globus pallidus, striatum, subthalamic nucleus, substantia nigra, thalamus, and motor cortex. This concept supports action selection, movement scaling, and the control of competing motor programs.",
  "Mediodorsal Thalamus": "Prefrontal Integration\nPrefrontal integration is the routing of information between thalamus and frontal cortex for cognition and decision-making. It involves mediodorsal thalamus, prefrontal cortex, orbitofrontal cortex, anterior cingulate cortex, basal ganglia, and limbic regions. This concept supports working memory, flexible control, value-based decision-making, and goal maintenance.",
  "Pulvinar": "Visual Attention Coordination\nVisual attention coordination is the regulation of information flow across visual and parietal cortices. It involves the pulvinar, visual cortex, posterior parietal cortex, temporal visual areas, superior colliculus, and frontal eye fields. This concept supports selective attention, distractor filtering, and coordination of visual processing across cortical regions.",
  "Midbrain": "Dopaminergic and Orienting Control\nDopaminergic and orienting control links reward, movement, arousal, and sensory orienting. It involves the midbrain, substantia nigra, ventral tegmental area, superior colliculus, periaqueductal gray, brainstem reticular formation, and thalamic connections. This concept supports reward learning, eye movements, alertness, and basic action readiness.",
  "Pons": "Sensorimotor and Arousal Relay\nSensorimotor and arousal relay supports communication among cortex, cerebellum, and brainstem systems. It involves the pons, pontine nuclei, reticular formation, cerebellar connections, cranial nerve nuclei, and ascending arousal pathways. This concept supports sleep-wake regulation, facial and eye movement control, breathing modulation, and coordination of sensorimotor signals.",
  "Medulla": "Autonomic Control\nAutonomic control is the regulation of vital bodily functions such as breathing, heart rate, and reflexive responses. It involves the medulla, nucleus tractus solitarius, ventrolateral medulla, dorsal motor vagal regions, reticular formation, and connections with hypothalamus and spinal cord. This concept supports respiratory rhythm, cardiovascular stability, swallowing, and homeostatic reflexes.",
  "Cerebellar Vermis": "Postural and Affective Coordination\nPostural and affective coordination links midline motor control with autonomic and emotional regulation. It involves the cerebellar vermis, fastigial nucleus, brainstem vestibular nuclei, limbic connections, hypothalamus, and medial cortical regions. This concept supports balance, gait stability, arousal modulation, and coordinated bodily responses to affective context.",
  "Cerebellar Hemispheres": "Cognitive and Motor Prediction\nCognitive and motor prediction is the anticipation and correction of errors across movement and thought. It involves the cerebellar hemispheres, dentate nucleus, thalamus, prefrontal cortex, parietal cortex, motor cortex, and cerebro-cerebellar loops. This concept supports skilled action, timing, language-related sequencing, working memory, and adaptive prediction.",
  "Dentate Nucleus": "Cerebellar Output Control\nCerebellar output control is the transmission of cerebellar computations to motor and cognitive cortical systems. It involves the dentate nucleus, cerebellar hemispheres, superior cerebellar peduncle, thalamus, motor cortex, prefrontal cortex, and parietal cortex. This concept supports coordination, timing, error correction, and the modulation of cortical planning systems.",
  "Primary Gustatory Cortex": "Taste Perception\nTaste perception is the cortical representation of sweet, salty, sour, bitter, umami, and oral sensory qualities. It involves primary gustatory cortex in the anterior insula and frontal operculum, with connections to orbitofrontal cortex, somatosensory cortex, amygdala, and hypothalamus. This concept supports flavor evaluation, food preference, and integration of taste with bodily needs.",
  "Olfactory Cortex": "Odor Perception\nOdor perception is the detection and interpretation of smells as meaningful sensory and emotional cues. It involves piriform cortex, orbitofrontal cortex, amygdala, entorhinal cortex, hippocampus, and olfactory bulb projections. This concept supports odor identification, flavor experience, emotional association, and memory-linked sensory recognition.",
  "Vestibular Cortex": "Balance and Self-Motion Perception\nBalance and self-motion perception represent head movement, orientation, and body position in space. They involve vestibular cortex in parietal operculum and posterior insula, temporoparietal regions, cerebellum, brainstem vestibular nuclei, and posterior parietal cortex. This concept supports balance, spatial orientation, gaze stabilization, and the sense of bodily motion.",
  "Memory Network": "Episodic Memory\nEpisodic memory is the ability to encode, retrieve, and reconstruct personally experienced events. It involves hippocampus, medial temporal lobe, posterior cingulate cortex, precuneus, retrosplenial cortex, angular gyrus, and medial prefrontal cortex. This concept supports remembering events, imagining future situations, and linking experiences across time and context.",
  "Speech Production Network": "Speech Production\nSpeech production is the transformation of linguistic intentions into articulated speech. It involves inferior frontal gyrus, premotor cortex, supplementary motor area, primary motor cortex for face and mouth, anterior insula, basal ganglia, cerebellum, and auditory feedback regions. This concept supports word formulation, phonological sequencing, articulation planning, and fluent verbal output.",
  "Speech Perception Network": "Speech Perception\nSpeech perception is the interpretation of acoustic speech signals as phonemes, words, and meaningful language. It involves superior temporal gyrus, superior temporal sulcus, primary auditory cortex, middle temporal gyrus, inferior frontal gyrus, and temporoparietal regions. This concept supports recognizing spoken words, parsing speech sounds, and mapping auditory input to meaning.",
  "Numerical Cognition Network": "Numerical Cognition\nNumerical cognition is the representation and manipulation of quantity, magnitude, and symbolic number. It involves intraparietal sulcus, inferior parietal cortex, superior parietal lobule, dorsolateral prefrontal cortex, angular gyrus, and visual-symbolic processing regions. This concept supports arithmetic, magnitude comparison, counting, and quantitative reasoning.",
  "Spatial Navigation Network": "Spatial Navigation\nSpatial navigation is the representation of location, direction, landmarks, and routes through environments. It involves hippocampus, entorhinal cortex, parahippocampal cortex, retrosplenial cortex, posterior parietal cortex, precuneus, and medial prefrontal cortex. This concept supports wayfinding, mental maps, route planning, and memory for spatial context.",
  "Hand Motor Cortex": "Hand Movement Control\nHand movement control is the execution and refinement of voluntary finger and hand actions. It involves the hand area of primary motor cortex, primary somatosensory cortex, premotor cortex, supplementary motor area, cerebellum, basal ganglia, and corticospinal pathways. This concept supports grasping, reaching, object manipulation, and fine motor coordination.",
  "Foot Motor Cortex": "Foot Movement Control\nFoot movement control is the execution and coordination of lower-limb actions. It involves the medial primary motor cortex near the paracentral lobule, primary somatosensory leg area, supplementary motor area, basal ganglia, cerebellum, and spinal motor pathways. This concept supports walking, balance-related movement, posture adjustment, and voluntary foot control.",
  "Face Motor Cortex": "Orofacial Movement Control\nOrofacial movement control is the coordination of facial, mouth, and jaw movements. It involves the lateral primary motor cortex, premotor cortex, supplementary motor area, anterior insula, basal ganglia, cerebellum, and cranial motor pathways. This concept supports facial expression, articulation, chewing, swallowing, and expressive communication.",
  "Paracentral Lobule": "Lower-Limb Sensorimotor Control\nLower-limb sensorimotor control is the integration of movement and sensation for the legs and feet. It involves the paracentral lobule, medial primary motor cortex, medial primary somatosensory cortex, supplementary motor area, cerebellum, and corticospinal pathways. This concept supports gait, posture, foot movement, and sensory feedback from the lower body.",
  "Frontal Operculum": "Speech and Salience Integration\nSpeech and salience integration links articulatory planning, interoceptive awareness, and task-relevant control. It involves the frontal operculum, anterior insula, inferior frontal gyrus, anterior cingulate cortex, premotor cortex, and superior temporal language regions. This concept supports speech planning, response control, taste and oral sensation, and detection of behaviorally relevant signals.",
  "Parietal Operculum": "Somatosensory Integration\nSomatosensory integration combines tactile, pain, vestibular, and body-related signals. It involves the parietal operculum, secondary somatosensory cortex, posterior insula, primary somatosensory cortex, thalamus, and vestibular cortical regions. This concept supports touch perception, body awareness, pain processing, and integration of sensory signals from both sides of the body.",
  "Cuneus": "Basic Visual Representation\nBasic visual representation is the encoding of visual field information and low-level visual features. It involves the cuneus, primary visual cortex, calcarine cortex, extrastriate occipital cortex, and dorsal visual regions. This concept supports visual awareness, spatially organized perception, and early analysis of visual input.",
  "Lingual Gyrus": "Visual Form and Color Processing\nVisual form and color processing is the analysis of visual patterns, letters, scenes, and chromatic features. It involves the lingual gyrus, calcarine cortex, fusiform gyrus, ventral occipital cortex, and parahippocampal visual regions. This concept supports reading-related visual processing, scene perception, color analysis, and recognition of structured visual forms.",
  "Precentral Gyrus": "Motor Execution\nMotor execution is the cortical initiation of voluntary movement commands. It involves the precentral gyrus, primary motor cortex, premotor cortex, supplementary motor area, corticospinal tract, basal ganglia, cerebellum, and thalamus. This concept supports somatotopic control of body movements, action output, and coordination of voluntary motor behavior.",
  "Postcentral Gyrus": "Body Sensory Mapping\nBody sensory mapping is the cortical representation of touch and proprioceptive input across the body surface. It involves the postcentral gyrus, primary somatosensory cortex, thalamic sensory nuclei, posterior parietal cortex, and secondary somatosensory cortex. This concept supports tactile discrimination, body localization, and sensory guidance of movement.",
  "Middle Frontal Gyrus": "Working Memory Control\nWorking memory control is the maintenance and manipulation of information for ongoing cognition. It involves the middle frontal gyrus, dorsolateral prefrontal cortex, inferior parietal cortex, anterior cingulate cortex, and basal ganglia. This concept supports goal maintenance, planning, response selection, and flexible problem solving.",
  "Superior Frontal Gyrus": "Goal Maintenance\nGoal maintenance is the sustained representation of intentions, rules, and internally guided plans. It involves the superior frontal gyrus, medial prefrontal cortex, supplementary motor area, dorsolateral prefrontal cortex, and frontoparietal control regions. This concept supports self-directed behavior, planning, attention control, and regulation of ongoing cognition.",
  "Inferior Temporal Gyrus": "Object and Semantic Recognition\nObject and semantic recognition is the identification of visual categories and their associated meanings. It involves inferior temporal gyrus, fusiform gyrus, lateral occipital cortex, anterior temporal cortex, and ventral visual stream regions. This concept supports object naming, category recognition, and linking visual perception with conceptual knowledge.",
  "Calcarine Cortex": "Primary Visual Encoding\nPrimary visual encoding is the retinotopic cortical representation of visual input. It involves calcarine cortex, primary visual cortex, lateral geniculate nucleus, cuneus, lingual gyrus, and nearby extrastriate regions. This concept supports early visual awareness, spatial precision, and extraction of basic visual features.",
  "Nucleus Accumbens": "Reward Motivation\nReward motivation is the drive to approach valued outcomes and learn from reinforcement. It involves the nucleus accumbens, ventral striatum, ventral tegmental area, orbitofrontal cortex, ventromedial prefrontal cortex, amygdala, and hippocampus. This concept supports incentive learning, reward anticipation, and motivation-guided behavior.",
  "Inferior Parietal Cortex": "Attention and Conceptual Integration\nAttention and conceptual integration combine spatial, linguistic, and memory-related information. It involves inferior parietal cortex, angular gyrus, supramarginal gyrus, posterior temporal cortex, intraparietal sulcus, and lateral frontal cortex. This concept supports attentional reorienting, semantic integration, body representation, and flexible reasoning.",
  "Parahippocampal Gyrus": "Contextual Memory\nContextual memory is the representation of scenes, places, and environmental associations. It involves the parahippocampal gyrus, parahippocampal cortex, hippocampus, entorhinal cortex, retrosplenial cortex, and posterior cingulate cortex. This concept supports place recognition, scene context, spatial memory, and linking experiences to environments.",
  "Frontal Pole": "Abstract Goal Management\nAbstract goal management is the representation of long-range plans, alternatives, and higher-order intentions. It involves the frontal pole, anterior prefrontal cortex, dorsolateral prefrontal cortex, frontoparietal control network, medial prefrontal cortex, and parietal association regions. This concept supports prospective thinking, multitasking, strategic planning, and evaluation of competing goals.",
  "Multiple Demand Network": "Domain-General Cognitive Demand\nDomain-general cognitive demand is the flexible recruitment of control systems during difficult tasks. It involves lateral prefrontal cortex, anterior insula, dorsal anterior cingulate cortex, intraparietal sulcus, inferior parietal lobule, and pre-supplementary motor area. This concept supports problem solving, task switching, working memory, and adaptive control across many cognitive domains.",
  "Salience/Ventral Attention Network": "Salience-Guided Reorienting\nSalience-guided reorienting is the detection of important events and the redirection of attention toward them. It involves anterior insula, dorsal anterior cingulate cortex, temporoparietal junction, frontal operculum, ventral frontal cortex, and subcortical salience-related regions. This concept supports interrupting ongoing processing, prioritizing relevant signals, and coordinating attention with control demands.",
  "Primary Visual Network": "Early Visual Processing\nEarly visual processing is the initial analysis of retinotopic visual input and basic image features. It involves primary visual cortex, calcarine cortex, cuneus, lingual gyrus, and adjacent extrastriate occipital areas. This concept supports visual awareness, edge and contrast detection, and the foundation for later recognition and spatial perception.",
  "Visual Association Network": "High-Level Visual Association\nHigh-level visual association is the interpretation of complex visual forms, categories, scenes, and motion. It involves extrastriate visual cortex, lateral occipital cortex, fusiform gyrus, inferior temporal cortex, parahippocampal visual regions, and dorsal parietal visual areas. This concept supports object recognition, scene understanding, visual categorization, and visual guidance of behavior.",
  "Primary Auditory Network": "Early Auditory Processing\nEarly auditory processing is the cortical encoding of basic acoustic properties. It involves primary auditory cortex in Heschl's gyrus, medial geniculate thalamus, superior temporal plane, and nearby auditory belt regions. This concept supports pitch perception, temporal sound analysis, and the foundation for speech, music, and environmental sound recognition.",
  "Auditory Association Network": "Meaningful Sound Processing\nMeaningful sound processing is the interpretation of complex auditory input as speech, voices, music, or environmental events. It involves superior temporal gyrus, superior temporal sulcus, planum temporale, middle temporal gyrus, auditory association cortex, and inferior frontal regions. This concept supports auditory comprehension, voice recognition, speech perception, and sound-based communication.",
  "Hand Sensorimotor Network": "Hand Sensorimotor Control\nHand sensorimotor control integrates tactile feedback with fine voluntary hand movement. It involves hand areas of primary motor cortex and primary somatosensory cortex, premotor cortex, posterior parietal cortex, basal ganglia, cerebellum, and thalamus. This concept supports grasping, manipulation, reaching, and precise coordination of fingers and hands.",
  "Foot Sensorimotor Network": "Foot Sensorimotor Control\nFoot sensorimotor control integrates lower-limb sensation with voluntary and postural movement. It involves medial primary motor and somatosensory cortices, paracentral lobule, supplementary motor area, cerebellum, basal ganglia, and spinal pathways. This concept supports gait, balance, foot placement, and coordinated lower-body movement.",
  "Mouth Sensorimotor Network": "Oromotor Control\nOromotor control coordinates mouth, tongue, jaw, and facial movements with sensory feedback. It involves lateral motor cortex, somatosensory mouth regions, premotor cortex, anterior insula, frontal operculum, basal ganglia, cerebellum, and cranial motor pathways. This concept supports speech articulation, chewing, swallowing, and expressive facial movement.",
  "Dorsal Language Stream": "Auditory-Motor Language Mapping\nAuditory-motor language mapping links heard speech with articulatory and phonological representations. It involves posterior superior temporal cortex, supramarginal gyrus, inferior frontal gyrus, premotor cortex, arcuate fasciculus, and sensorimotor speech regions. This concept supports speech repetition, phonological working memory, articulation planning, and mapping sounds to spoken output.",
  "Ventral Language Stream": "Speech Meaning Mapping\nSpeech meaning mapping links auditory or visual language input to semantic comprehension. It involves superior temporal cortex, middle temporal gyrus, anterior temporal cortex, inferior frontal gyrus, angular gyrus, and ventral temporal language pathways. This concept supports word comprehension, sentence meaning, semantic association, and understanding language content.",
  "Speech Network": "Spoken Language Processing\nSpoken language processing includes perceiving, planning, and producing speech. It involves superior temporal gyrus, inferior frontal gyrus, premotor cortex, supplementary motor area, anterior insula, auditory cortex, basal ganglia, and cerebellum. This concept supports speech comprehension, articulation, phonological sequencing, and fluent verbal communication.",
  "Affective Network": "Affective Processing\nAffective processing is the evaluation and regulation of emotional significance. It involves amygdala, anterior insula, anterior cingulate cortex, orbitofrontal cortex, ventromedial prefrontal cortex, hypothalamus, ventral striatum, and brainstem. This concept supports emotional appraisal, bodily feeling states, motivation, and adaptive responses to meaningful events.",
  "Orbitofrontal-Affective Network": "Affective Value Evaluation\nAffective value evaluation is the integration of sensory, emotional, and reward information into subjective preferences. It involves orbitofrontal cortex, ventromedial prefrontal cortex, amygdala, ventral striatum, anterior insula, hypothalamus, and temporal association regions. This concept supports preference formation, outcome evaluation, emotional decision-making, and flexible updating of value.",
  "Medial Temporal Memory Network": "Declarative Memory\nDeclarative memory is the encoding and retrieval of facts, events, and relational associations. It involves hippocampus, entorhinal cortex, perirhinal cortex, parahippocampal cortex, retrosplenial cortex, posterior cingulate cortex, and medial prefrontal cortex. This concept supports remembering experiences, recognizing familiarity, and linking information across time and context.",
  "Hippocampal Network": "Relational Episodic Memory\nRelational episodic memory is the binding of people, places, objects, and temporal context into retrievable events. It involves hippocampus, entorhinal cortex, parahippocampal cortex, retrosplenial cortex, posterior cingulate cortex, medial prefrontal cortex, and angular gyrus. This concept supports event memory, spatial context, mental simulation, and flexible recombination of past experiences.",
  "Posterior Medial Network": "Contextual Integration\nContextual integration is the construction of event, scene, and self-relevant context from memory and perception. It involves posterior cingulate cortex, retrosplenial cortex, precuneus, parahippocampal cortex, angular gyrus, hippocampus, and medial prefrontal cortex. This concept supports scene construction, autobiographical memory, spatial context, and internally directed thought.",
  "Object Perception Network": "Object Perception\nObject perception is the recognition of visual forms as stable objects and categories. It involves lateral occipital cortex, ventral occipitotemporal cortex, fusiform gyrus, inferior temporal cortex, anterior temporal cortex, and early visual cortex. This concept supports identifying objects, distinguishing categories, and linking visual appearance with semantic knowledge.",
  "Theory of Mind Network": "Mental State Reasoning\nMental state reasoning is the inference of beliefs, intentions, perspectives, and emotions in other people. It involves medial prefrontal cortex, temporoparietal junction, posterior superior temporal sulcus, precuneus, posterior cingulate cortex, and temporal poles. This concept supports social understanding, perspective taking, and prediction of behavior from inferred mental states.",
  "Social Cognition Network": "Social Cognition\nSocial cognition is the interpretation of people, actions, intentions, emotions, and social relationships. It involves medial prefrontal cortex, temporoparietal junction, superior temporal sulcus, temporal pole, amygdala, anterior insula, and posterior cingulate cortex. This concept supports understanding others, evaluating social cues, and guiding behavior in interpersonal contexts.",
  "Interoceptive Network": "Interoceptive Processing\nInteroceptive processing is the representation of internal bodily states and their affective significance. It involves posterior insula, anterior insula, anterior cingulate cortex, somatosensory cortex, hypothalamus, brainstem, and thalamus. This concept supports bodily awareness, emotional feeling, homeostatic regulation, and integration of internal state with decision-making.",
  "Oculomotor Network": "Gaze Control\nGaze control is the coordination of eye movements with attention and visual goals. It involves frontal eye fields, supplementary eye fields, intraparietal sulcus, superior colliculus, cerebellum, basal ganglia, and visual cortex. This concept supports saccades, smooth pursuit, visual search, and orienting toward relevant information.",
  "Visuospatial Network": "Visuospatial Cognition\nVisuospatial cognition is the representation and manipulation of spatial relationships in visual and body-centered coordinates. It involves posterior parietal cortex, intraparietal sulcus, superior parietal lobule, occipital visual cortex, frontal eye fields, precuneus, and dorsal visual stream regions. This concept supports spatial attention, mental rotation, navigation, reaching, and visual guidance of action.",
  "Basal Ganglia Network": "Action and Reinforcement Control\nAction and reinforcement control coordinates movement, habit, reward learning, and response selection. It involves striatum, caudate nucleus, putamen, globus pallidus, subthalamic nucleus, substantia nigra, thalamus, and frontal cortex. This concept supports selecting actions, learning from feedback, forming habits, and regulating cognitive and motor loops.",
  "Fronto-Striatal Network": "Goal-Directed Control\nGoal-directed control is the regulation of behavior through interactions between frontal goals and striatal action-selection systems. It involves prefrontal cortex, anterior cingulate cortex, orbitofrontal cortex, caudate nucleus, putamen, ventral striatum, globus pallidus, and thalamus. This concept supports planning, reinforcement learning, inhibition, working memory, and adaptive choice.",
  "Thalamic Network": "Cortical Communication\nCortical communication is the routing and modulation of information among sensory, motor, limbic, and association systems. It involves thalamic nuclei such as mediodorsal thalamus, pulvinar, ventral posterior nuclei, anterior thalamus, lateral geniculate nucleus, and widespread cortical targets. This concept supports attention, arousal, sensory relay, memory circuits, and flexible coordination across brain networks.",
  "Cerebellar Network": "Predictive Coordination\nPredictive coordination is the timing, calibration, and error-based adjustment of movement and cognition. It involves cerebellar cortex, cerebellar vermis, cerebellar hemispheres, dentate nucleus, deep cerebellar nuclei, brainstem, thalamus, and cortical motor and association regions. This concept supports motor coordination, sequence learning, timing, and adaptive prediction across behavior.",
  "Cerebro-Cerebellar Network": "Cerebro-Cerebellar Prediction\nCerebro-cerebellar prediction is the interaction between cerebral cortex and cerebellum for timing, error correction, and adaptive control. It involves cerebellar hemispheres, dentate nucleus, thalamus, prefrontal cortex, parietal cortex, motor cortex, premotor cortex, and pontine-cerebellar pathways. This concept supports coordinated movement, cognitive sequencing, working memory modulation, and predictive adjustment of ongoing behavior."

}


# %%
df_pubs_l = pd.read_parquet(TEXT_SYNTH_LESS_PATH)
df_pubs_l["text"] = df_pubs_l["title"].astype(str) + "\n" + df_pubs_l["description"].astype(str)
df_pubs_l["title_canonical"] = [canonical_title[i] for i in df_pubs_l["text"].str.split("\n").str[0]]
df_pubs_l = df_pubs_l[df_pubs_l["text"].str.len() <= 550]
df_pubs_l['label'] = [concept_level[i].upper() for i in df_pubs_l["title"]]

df_pubs_l['desc'] = df_pubs_l["text"].str.split("\n").str[1:].str[0]
df_pubs_l['text'] = df_pubs_l["title_canonical"] + "\n" + df_pubs_l["desc"]

# # Concept labels
df_pubs_l['text'] = ['[' + concept_level[i].upper() + ']' for i in df_pubs_l["title"]] + df_pubs_l["text"]

df_pubs = df_pubs_l.copy()
df_pubs.reset_index(inplace=True, drop=True)


df_func = (
    [
        {
        "network": "Visual Network",
        "title": "[FUNCTION]Visual Perception",
        "description": "Visual perception is the interpretation of visual information into coherent representations of the external world. It is most closely associated with primary visual cortex along the calcarine sulcus, extrastriate occipital cortex, lateral occipital cortex, fusiform gyrus, and dorsal occipitoparietal visual areas. This function supports the perception of form, color, spatial layout, and visually guided understanding of the environment."
        },
        {
        "network": "Visual Network",
        "title": "[FUNCTION]Early Visual Processing",
        "description": "Early visual processing is the extraction of basic visual features from incoming sensory input. It is centered on primary visual cortex, pericalcarine cortex, cuneus, lingual gyrus, and nearby extrastriate occipital regions. This function supports detection of edges, contrast, orientation, retinotopic position, and basic visual structure."
        },
        {
        "network": "Visual Network",
        "title": "[FUNCTION]Object Perception",
        "description": "Object perception is the ability to identify coherent visual forms as meaningful objects. It involves lateral occipital cortex, ventral occipitotemporal cortex, posterior inferior temporal cortex, and fusiform gyrus. This function supports recognition of shapes, object categories, and stable visual identities across changes in viewpoint or lighting."
        },
        {
        "network": "Visual Network",
        "title": "[FUNCTION]Face Perception",
        "description": "Face perception is the visual analysis of facial structure and identity-relevant features. It is associated with fusiform face-selective cortex, inferior occipital cortex, posterior superior temporal sulcus, and adjacent ventral temporal visual areas. This function supports recognition of facial form, facial identity cues, and socially relevant visual features."
        },
        {
        "network": "Visual Network",
        "title": "[FUNCTION]Motion Perception",
        "description": "Motion perception is the detection and interpretation of movement in the visual field. It is associated with lateral occipitotemporal cortex, area MT/V5, dorsal occipital cortex, and posterior parietal visual regions. This function supports perception of object motion, biological movement, visual flow, and visually guided tracking."
        },
        {
        "network": "Visual Network",
        "title": "[FUNCTION]Spatial Vision",
        "description": "Spatial vision is the representation of where visual stimuli are located in relation to the observer and the surrounding scene. It involves dorsal occipital cortex, superior parietal lobule, intraparietal sulcus, and occipitoparietal visual association cortex. This function supports spatial localization, visual attention across space, and visually guided action."
        },

        {
        "network": "Somatomotor Network",
        "title": "[FUNCTION]Sensorimotor Control",
        "description": "Sensorimotor control is the coordination of body sensation with voluntary movement. It is associated with precentral gyrus, postcentral gyrus, supplementary motor area, premotor cortex, paracentral lobule, basal ganglia, cerebellum, and motor thalamus. This function supports coordinated movement, body-state monitoring, and adjustment of actions based on sensory feedback."
        },
        {
        "network": "Somatomotor Network",
        "title": "[FUNCTION]Somatosensory Processing",
        "description": "Somatosensory processing is the interpretation of tactile and bodily sensory input. It is centered on primary somatosensory cortex in the postcentral gyrus, secondary somatosensory cortex near the parietal operculum, posterior insula, superior parietal lobule, and thalamic somatosensory nuclei. This function supports perception of touch, pressure, vibration, pain-related signals, and body surface location."
        },
        {
        "network": "Somatomotor Network",
        "title": "[FUNCTION]Motor Execution",
        "description": "Motor execution is the generation of voluntary body movements. It is associated with primary motor cortex in the precentral gyrus, supplementary motor area, premotor cortex, corticospinal motor regions, basal ganglia, cerebellum, and ventrolateral thalamus. This function supports controlled movement of the limbs, face, trunk, and speech-related musculature."
        },
        {
        "network": "Somatomotor Network",
        "title": "[FUNCTION]Motor Planning",
        "description": "Motor planning is the preparation and organization of intended movements before execution. It is associated with premotor cortex, supplementary motor area, dorsal precentral regions, superior parietal lobule, basal ganglia, and cerebellar motor territories. This function supports selection of movement sequences, preparation of body actions, and coordination of planned motor output."
        },
        {
        "network": "Somatomotor Network",
        "title": "[FUNCTION]Proprioception",
        "description": "Proprioception is the sensing of body position, joint state, and movement of the limbs. It is associated with postcentral gyrus, superior parietal lobule, paracentral lobule, cerebellum, thalamic somatosensory nuclei, and posterior insula. This function supports body awareness, movement calibration, posture, and online correction of actions."
        },

        {
        "network": "Auditory Network",
        "title": "[FUNCTION]Auditory Perception",
        "description": "Auditory perception is the interpretation of sound information from the environment. It is centered on Heschl’s gyrus, planum temporale, superior temporal gyrus, superior temporal sulcus, and auditory association cortex. This function supports perception of pitch, loudness, timbre, rhythm, and meaningful acoustic patterns."
        },
        {
        "network": "Auditory Network",
        "title": "[FUNCTION]Early Auditory Processing",
        "description": "Early auditory processing is the extraction of basic acoustic features from sound. It is associated with primary auditory cortex in Heschl’s gyrus, adjacent superior temporal plane, medial geniculate thalamus, and posterior superior temporal cortex. This function supports detection of frequency, intensity, timing, and basic sound structure."
        },
        {
        "network": "Auditory Network",
        "title": "[FUNCTION]Speech Sound Perception",
        "description": "Speech sound perception is the analysis of spoken acoustic input as phonological information. It is associated with bilateral superior temporal gyrus, posterior superior temporal sulcus, planum temporale, and left posterior temporal auditory-language cortex. This function supports recognition of speech sounds, syllable structure, and spoken language input."
        },
        {
        "network": "Auditory Network",
        "title": "[FUNCTION]Sound Localization",
        "description": "Sound localization is the estimation of where sounds originate in external space. It involves posterior superior temporal cortex, planum temporale, inferior parietal cortex, auditory association cortex, and subcortical auditory pathways. This function supports spatial hearing, orientation toward sound sources, and integration of auditory information with spatial attention."
        },
        {
        "network": "Auditory Network",
        "title": "[FUNCTION]Auditory Scene Analysis",
        "description": "Auditory scene analysis is the organization of complex sound mixtures into distinct perceptual sources. It is associated with superior temporal gyrus, planum temporale, superior temporal sulcus, inferior frontal auditory-control regions, and auditory association cortex. This function supports separation of speech, music, environmental sounds, and background noise into meaningful auditory objects."
        },
        {
        "network": "Language Network",
        "title": "[FUNCTION]Language Comprehension",
        "description": "Language comprehension is the interpretation of spoken or written language into meaning. It is associated with left posterior superior temporal gyrus, posterior superior temporal sulcus, middle temporal gyrus, angular gyrus, inferior frontal gyrus, and anterior temporal cortex. This function supports understanding of words, sentences, narrative structure, and linguistic meaning."
        },
        {
        "network": "Language Network",
        "title": "[FUNCTION]Semantic Processing",
        "description": "Semantic processing is the representation and retrieval of conceptual meaning from language. It is associated with left middle temporal gyrus, anterior temporal lobe, angular gyrus, inferior frontal gyrus, posterior superior temporal sulcus, and ventral temporal association cortex. This function supports word meaning, conceptual associations, and integration of meaning across phrases and sentences."
        },
        {
        "network": "Language Network",
        "title": "[FUNCTION]Syntactic Processing",
        "description": "Syntactic processing is the organization of words into structured grammatical relationships. It is associated with left inferior frontal gyrus, posterior superior temporal sulcus, posterior middle temporal gyrus, and temporoparietal language regions. This function supports sentence structure, grammatical dependencies, and interpretation of who did what to whom."
        },
        {
        "network": "Language Network",
        "title": "[FUNCTION]Phonological Processing",
        "description": "Phonological processing is the representation and manipulation of speech-sound structure. It is associated with left superior temporal gyrus, planum temporale, posterior superior temporal sulcus, supramarginal gyrus, and inferior frontal gyrus. This function supports speech-sound recognition, syllable structure, verbal working memory, and sound-to-word mapping."
        },
        {
        "network": "Language Network",
        "title": "[FUNCTION]Speech Production",
        "description": "Speech production is the planning and generation of spoken language. It is associated with left inferior frontal gyrus, ventral premotor cortex, supplementary motor area, anterior insula, superior temporal auditory-feedback regions, and motor cortex for orofacial articulation. This function supports word selection, articulatory planning, fluent speech output, and monitoring of produced speech."
        },
        {
        "network": "Language Network",
        "title": "[FUNCTION]Reading Comprehension",
        "description": "Reading comprehension is the extraction of linguistic meaning from written text. It is associated with left occipitotemporal visual word-form regions, posterior superior temporal cortex, middle temporal gyrus, angular gyrus, and inferior frontal gyrus. This function supports mapping visual word forms to sounds, meanings, sentence structure, and conceptual interpretation."
        },

        {
        "network": "Dorsal Attention Network",
        "title": "[FUNCTION]Goal-Directed Attention",
        "description": "Goal-directed attention is the voluntary allocation of attention toward behaviorally relevant information. It is associated with intraparietal sulcus, superior parietal lobule, frontal eye fields, dorsal premotor cortex, and extrastriate visual cortex. This function supports top-down selection of locations, features, and objects according to current goals."
        },
        {
        "network": "Dorsal Attention Network",
        "title": "[FUNCTION]Visuospatial Attention",
        "description": "Visuospatial attention is the selective prioritization of locations in visual space. It is associated with bilateral intraparietal sulcus, superior parietal lobule, frontal eye fields, dorsal occipital cortex, and posterior parietal cortex. This function supports spatial orienting, visual search, and enhancement of relevant locations in the visual field."
        },
        {
        "network": "Dorsal Attention Network",
        "title": "[FUNCTION]Attentional Orienting",
        "description": "Attentional orienting is the shifting of attention toward selected locations or features. It is associated with frontal eye fields, intraparietal sulcus, superior parietal lobule, dorsal premotor cortex, and visual association cortex. This function supports preparation for relevant sensory input and flexible redirection of attention across space."
        },
        {
        "network": "Dorsal Attention Network",
        "title": "[FUNCTION]Visual Search",
        "description": "Visual search is the active scanning of the visual environment to locate relevant targets. It is associated with intraparietal sulcus, superior parietal lobule, frontal eye fields, lateral occipital cortex, and dorsal visual association areas. This function supports efficient selection of relevant stimuli among competing visual information."
        },

        {
        "network": "Ventral Attention Network",
        "title": "[FUNCTION]Stimulus-Driven Attention",
        "description": "Stimulus-driven attention is the reorientation of attention toward unexpected or behaviorally relevant events. It is associated with right temporoparietal junction, ventral frontal cortex, inferior frontal gyrus, frontal operculum, and anterior insula. This function supports detection of relevant changes in the environment and interruption of ongoing attentional focus."
        },
        {
        "network": "Ventral Attention Network",
        "title": "[FUNCTION]Attentional Reorienting",
        "description": "Attentional reorienting is the redirection of attention when new information becomes relevant. It is associated with right temporoparietal junction, inferior frontal junction, ventral frontal cortex, anterior insula, and middle frontal regions. This function supports flexible updating of attentional priorities when expectations or environmental conditions change."
        },
        {
        "network": "Ventral Attention Network",
        "title": "[FUNCTION]Unexpected Stimulus Detection",
        "description": "Unexpected stimulus detection is the identification of salient events that were not the current focus of attention. It is associated with temporoparietal junction, ventral frontal cortex, frontal operculum, anterior insula, and lateral temporal-parietal association cortex. This function supports rapid updating of attention toward novel, surprising, or behaviorally relevant cues."
        },

        {
        "network": "Salience Network",
        "title": "[FUNCTION]Salience Detection",
        "description": "Salience detection is the identification of stimuli or internal signals that are important for guiding attention and behavior. It is associated with anterior insula, frontal operculum, dorsal anterior cingulate cortex, anterior midcingulate cortex, and subcortical nodes including thalamus and ventral striatum. This function supports prioritization of relevant information and flexible engagement of attention and control systems."
        },
        {
        "network": "Salience Network",
        "title": "[FUNCTION]Interoceptive Attention",
        "description": "Interoceptive attention is the monitoring of internal bodily signals and their relevance to ongoing cognition. It is associated with anterior insula, mid-insula, dorsal anterior cingulate cortex, anterior midcingulate cortex, and somatosensory-interoceptive association regions. This function supports awareness of internal state, bodily feeling, and integration of physiological signals with attention."
        },
        {
        "network": "Salience Network",
        "title": "[FUNCTION]Control Network Switching",
        "description": "Control network switching is the coordination of transitions between internally oriented and externally oriented cognitive states. It is associated with anterior insula, frontal operculum, dorsal anterior cingulate cortex, anterior midcingulate cortex, and lateral prefrontal control regions. This function supports flexible recruitment of task-control systems when salient information requires a change in cognitive focus."
        },
        {
        "network": "Cingulo-Opercular Network",
        "title": "[FUNCTION]Sustained Task Set",
        "description": "Sustained task set is the maintenance of stable task goals across time. It is associated with dorsal anterior cingulate cortex, anterior midcingulate cortex, bilateral anterior insula, frontal operculum, anterior prefrontal cortex, and thalamic control regions. This function supports stable performance, maintenance of current task rules, and ongoing readiness to respond."
        },
        {
        "network": "Salience Network",
        "title": "[FUNCTION]Conflict Monitoring",
        "description": "Conflict monitoring is the detection of competition between incompatible actions, goals, or sources of information. It is associated with dorsal anterior cingulate cortex, anterior midcingulate cortex, anterior insula, frontal operculum, and lateral prefrontal control regions. This function supports recognition of situations that require increased cognitive control or adjustment of behavior."
        },

        {
        "network": "Frontoparietal Control Network",
        "title": "[FUNCTION]Cognitive Control",
        "description": "Cognitive control is the regulation of thought and behavior according to current goals. It is associated with dorsolateral prefrontal cortex, rostrolateral prefrontal cortex, inferior parietal lobule, intraparietal sulcus, anterior inferior parietal cortex, and lateral cerebellar control regions. This function supports flexible rule use, goal maintenance, planning, and adjustment of behavior."
        },
        {
        "network": "Frontoparietal Control Network",
        "title": "[FUNCTION]Working Memory",
        "description": "Working memory is the temporary maintenance and manipulation of information for ongoing cognition. It is associated with dorsolateral prefrontal cortex, inferior parietal lobule, intraparietal sulcus, lateral premotor cortex, and posterior parietal association cortex. This function supports holding information in mind, updating mental representations, and using stored information to guide behavior."
        },
        {
        "network": "Frontoparietal Control Network",
        "title": "[FUNCTION]Task Switching",
        "description": "Task switching is the flexible transition between different goals, rules, or cognitive operations. It is associated with dorsolateral prefrontal cortex, inferior parietal lobule, intraparietal sulcus, anterior prefrontal cortex, and lateral frontal control regions. This function supports adaptation when task demands change and helps coordinate behavior across competing rules."
        },
        {
        "network": "Frontoparietal Control Network",
        "title": "[FUNCTION]Goal-Directed Behavior",
        "description": "Goal-directed behavior is the organization of actions according to internal goals and contextual demands. It is associated with dorsolateral prefrontal cortex, rostrolateral prefrontal cortex, inferior parietal lobule, posterior parietal cortex, and lateral cerebellar association regions. This function supports planning, selection of relevant information, and adjustment of behavior toward intended outcomes."
        },
        {
        "network": "Frontoparietal Control Network",
        "title": "[FUNCTION]Decision Control",
        "description": "Decision control is the use of cognitive rules and goals to guide choices among alternatives. It is associated with dorsolateral prefrontal cortex, inferior parietal lobule, anterior prefrontal cortex, lateral frontal cortex, and posterior parietal association cortex. This function supports comparison of options, selection of task-relevant responses, and flexible updating of decisions."
        },
        {
        "network": "Frontoparietal Control Network",
        "title": "[FUNCTION]Executive Attention",
        "description": "Executive attention is the controlled allocation of cognitive resources toward goal-relevant information. It is associated with dorsolateral prefrontal cortex, inferior parietal lobule, intraparietal sulcus, lateral frontal cortex, and anterior prefrontal cortex. This function supports selective focus, maintenance of task goals, and suppression of distracting information."
        },

        {
        "network": "Default Mode Network",
        "title": "[FUNCTION]Self-Referential Thought",
        "description": "Self-referential thought is the evaluation of information in relation to oneself. It is associated with medial prefrontal cortex, posterior cingulate cortex, precuneus, angular gyrus, and ventral anterior medial frontal regions. This function supports reflection on personal traits, preferences, goals, and internally generated self-related content."
        },
        {
        "network": "Default Mode Network",
        "title": "[FUNCTION]Autobiographical Memory",
        "description": "Autobiographical memory is the retrieval and organization of personally experienced events. It is associated with posterior cingulate cortex, precuneus, medial prefrontal cortex, hippocampal formation, parahippocampal cortex, angular gyrus, and lateral temporal cortex. This function supports recollection of personal experiences and integration of memory with self-related context."
        },
        {
        "network": "Default Mode Network",
        "title": "[FUNCTION]Episodic Future Thinking",
        "description": "Episodic future thinking is the construction of possible future events from memory and imagination. It is associated with medial prefrontal cortex, posterior cingulate cortex, precuneus, hippocampal formation, parahippocampal cortex, angular gyrus, and lateral temporal cortex. This function supports simulation of future situations, planning of personal events, and internally generated scene construction."
        },
        {
        "network": "Default Mode Network",
        "title": "[FUNCTION]Internal Mentation",
        "description": "Internal mentation is the generation of thought that is oriented toward internal representations rather than immediate sensory input. It is associated with medial prefrontal cortex, posterior cingulate cortex, precuneus, angular gyrus, lateral temporal cortex, and medial temporal lobe. This function supports mind-wandering, reflective thought, memory-based cognition, and construction of internally guided mental content."
        },
        {
        "network": "Default Mode Network",
        "title": "[FUNCTION]Social Cognition",
        "description": "Social cognition is the representation of other people, social relationships, and socially meaningful information. It is associated with medial prefrontal cortex, temporoparietal junction, posterior cingulate cortex, precuneus, angular gyrus, superior temporal sulcus, and lateral temporal cortex. This function supports reasoning about people, social context, interpersonal meaning, and internally modeled social information."
        },
        {
        "network": "Default Mode Network",
        "title": "[FUNCTION]Theory of Mind",
        "description": "Theory of mind is the inference of others’ beliefs, intentions, and mental states. It is associated with medial prefrontal cortex, temporoparietal junction, posterior superior temporal sulcus, posterior cingulate cortex, precuneus, and temporal pole. This function supports interpretation of social behavior through internally represented mental-state models."
        },
        {
        "network": "Default Mode Network",
        "title": "[FUNCTION]Semantic Memory",
        "description": "Semantic memory is the representation of general conceptual knowledge. It is associated with lateral temporal cortex, angular gyrus, anterior temporal lobe, medial prefrontal cortex, posterior cingulate cortex, and inferior parietal regions. This function supports access to concepts, meanings, categories, and knowledge that is not tied to a single sensory event."
        },


        {
        "network": "Limbic Network",
        "title": "[FUNCTION]Affective Appraisal",
        "description": "Affective appraisal is the evaluation of the personal or motivational significance of stimuli. It is associated with orbitofrontal cortex, ventromedial prefrontal cortex, temporal pole, amygdala, anterior temporal cortex, and ventral striatum. This function supports assignment of emotional and motivational value to sensory, social, and internal information."
        },
        {
        "network": "Limbic Network",
        "title": "[FUNCTION]Reward Valuation",
        "description": "Reward valuation is the estimation of the subjective value of possible outcomes. It is associated with ventromedial prefrontal cortex, orbitofrontal cortex, ventral striatum, amygdala, and anterior temporal-limbic regions. This function supports preference formation, value comparison, and motivation-guided choice."
        },
        {
        "network": "Limbic Network",
        "title": "[FUNCTION]Emotion Appraisal",
        "description": "Emotion appraisal is the interpretation of stimuli in terms of emotional relevance. It is associated with amygdala, orbitofrontal cortex, ventromedial prefrontal cortex, anterior temporal cortex, temporal pole, and anterior insula. This function supports evaluation of affective meaning and integration of emotion with perception and decision-making."
        },
        {
        "network": "Limbic Network",
        "title": "[FUNCTION]Motivational Valuation",
        "description": "Motivational valuation is the representation of how strongly a stimulus or goal should influence behavior. It is associated with ventral striatum, orbitofrontal cortex, ventromedial prefrontal cortex, amygdala, hypothalamus, and anterior temporal-limbic cortex. This function supports approach behavior, preference-guided action, and prioritization of motivationally relevant information."
        },

        {
        "network": "Medial Temporal Memory Network",
        "title": "[FUNCTION]Episodic Memory Retrieval",
        "description": "Episodic memory retrieval is the recovery of specific past experiences from memory. It is associated with hippocampal formation, parahippocampal cortex, retrosplenial cortex, posterior cingulate cortex, precuneus, angular gyrus, and medial prefrontal cortex. This function supports recollection of events, contextual details, and memory-guided construction of internal scenes."
        },
        {
        "network": "Medial Temporal Memory Network",
        "title": "[FUNCTION]Contextual Memory",
        "description": "Contextual memory is the representation of where, when, and under what circumstances information was encountered. It is associated with hippocampus, parahippocampal cortex, retrosplenial cortex, posterior cingulate cortex, medial prefrontal cortex, and angular gyrus. This function supports linking details into coherent event contexts and using context to guide recall."
        },
        {
        "network": "Medial Temporal Memory Network",
        "title": "[FUNCTION]Scene Construction",
        "description": "Scene construction is the assembly of spatially coherent mental representations of places or events. It is associated with hippocampus, parahippocampal place-related cortex, retrosplenial cortex, posterior cingulate cortex, precuneus, and medial prefrontal cortex. This function supports imagination of environments, memory-based spatial context, and internally generated event simulations."
        },

        {
        "network": "Subcortical Network",
        "title": "[FUNCTION]Action Selection",
        "description": "Action selection is the process of choosing among competing behavioral responses. It is associated with dorsal striatum, caudate, putamen, globus pallidus, thalamus, supplementary motor area, premotor cortex, and lateral prefrontal cortex. This function supports selection, initiation, and regulation of actions according to current goals and context."
        },
        {
        "network": "Subcortical Network",
        "title": "[FUNCTION]Motor Gating",
        "description": "Motor gating is the regulation of which motor plans are facilitated or suppressed. It is associated with putamen, caudate, globus pallidus, subthalamic nucleus, thalamus, supplementary motor area, and primary motor cortex. This function supports controlled initiation of movement and suppression of competing motor outputs."
        },
        {
        "network": "Subcortical Network",
        "title": "[FUNCTION]Reinforcement Learning",
        "description": "Reinforcement learning is the updating of behavior based on reward, feedback, and prediction error. It is associated with ventral striatum, caudate, putamen, orbitofrontal cortex, ventromedial prefrontal cortex, midbrain dopaminergic regions, and thalamus. This function supports learning from outcomes and adjusting future choices based on value."
        },
        {
        "network": "Thalamic Network",
        "title": "[FUNCTION]Thalamocortical Integration",
        "description": "Thalamocortical integration is the coordination of information flow between thalamic nuclei and distributed cortical systems. It is associated with mediodorsal thalamus, pulvinar, ventrolateral thalamus, sensory thalamic nuclei, prefrontal cortex, parietal cortex, and sensory cortices. This function supports regulation of sensory processing, attention, motor control, and large-scale cortical communication."
        },

        {
        "network": "Cerebellar Network",
        "title": "[FUNCTION]Motor Coordination",
        "description": "Motor coordination is the fine adjustment of movement timing, precision, and sequencing. It is associated with anterior cerebellum, cerebellar lobules IV–VI, sensorimotor cerebellar territories, dentate nucleus, motor cortex, premotor cortex, and somatosensory cortex. This function supports smooth movement, timing of motor output, and correction of movement based on feedback."
        },
        {
        "network": "Cerebellar Network",
        "title": "[FUNCTION]Action Timing",
        "description": "Action timing is the temporal coordination of movements and predicted sensory consequences. It is associated with cerebellar lobules V–VI, lobule VIII, dentate nucleus, supplementary motor area, premotor cortex, and basal ganglia. This function supports rhythmic movement, temporal prediction, and coordination of actions over time."
        },
        {
        "network": "Cerebellar Network",
        "title": "[FUNCTION]Cognitive Sequencing",
        "description": "Cognitive sequencing is the organization of ordered mental operations across time. It is associated with posterior cerebellar lobules Crus I and Crus II, dentate nucleus, dorsolateral prefrontal cortex, inferior parietal lobule, and frontoparietal control regions. This function supports ordered planning, rule-based cognition, and coordination of multi-step mental processes."
        }
    ]
)
df_func = pd.DataFrame([i['title'] + "\n" + i["description"] for i in df_func], columns=['text'])
df_func_2 = pd.DataFrame(["[FUNCTION]" + i for i in function_desc.values()], columns=["text"])
df_more = pd.DataFrame([
    {
        'title': '[NETWORK]Default Mode Network',
        'description': 'The default mode network is a canonical fMRI network associated with internally directed cognition. Core regions include medial prefrontal cortex, posterior cingulate cortex, precuneus, angular gyrus, lateral temporal cortex, and hippocampal formation. It supports self-referential thought, autobiographical memory, social inference, and spontaneous internally generated cognition.'
    },
    {
        'title': '[NETWORK]Frontoparietal Control Network',
        'description': 'The frontoparietal control network is a canonical control network often described as the central executive network. Core regions include dorsolateral prefrontal cortex, lateral frontopolar cortex, inferior parietal lobule, and intraparietal sulcus. It supports flexible cognitive control, working memory, rule maintenance, and goal-directed behavior.'
    },
    {
        'title': '[NETWORK]Salience Network',
        'description': 'The salience network is a canonical fMRI network involved in detecting behaviorally relevant internal and external events. Core regions include anterior insula, dorsal anterior cingulate cortex, frontal operculum, and presupplementary motor area. It supports salience detection, interoceptive awareness, conflict monitoring, and switching between cognitive states.'
    },
    {
        'title': '[NETWORK]Dorsal Attention Network',
        'description': 'The dorsal attention network is a canonical fMRI network involved in voluntary spatial attention. Core regions include frontal eye fields, intraparietal sulcus, superior parietal lobule, and dorsal occipital cortex. It supports goal-directed orienting, visuospatial selection, and top-down allocation of attention.'
    },
    {
        'title': '[NETWORK]Ventral Attention Network',
        'description': 'The ventral attention network is a canonical fMRI network involved in stimulus-driven attentional reorienting. Core regions include temporoparietal junction, ventral frontal cortex, inferior frontal gyrus, and middle frontal gyrus. It supports detection of unexpected relevant stimuli and redirection of attention toward important events.'
    },
    {
        'title': '[NETWORK]Somatomotor Network',
        'description': 'The somatomotor network is a canonical fMRI network involved in movement and bodily sensation. Core regions include precentral gyrus, postcentral gyrus, supplementary motor area, paracentral lobule, and central operculum. It supports motor execution, somatosensory perception, body representation, and sensorimotor coordination.'
    },
    {
        'title': '[NETWORK]Visual Network',
        'description': 'The visual network is a canonical fMRI network involved in processing visual input. Core regions include primary visual cortex, extrastriate occipital cortex, lateral occipital cortex, lingual gyrus, fusiform gyrus, and middle temporal visual area. It supports perception of form, color, motion, object structure, and spatial visual features.'
    },
    {
        'title': '[NETWORK]Auditory Network',
        'description': 'The auditory network is a common fMRI network involved in processing sound. Core regions include Heschl gyrus, planum temporale, superior temporal gyrus, and superior temporal sulcus. It supports auditory perception, speech sound analysis, temporal acoustic processing, and sound-based communication.'
    },
    {
        'title': '[NETWORK]Language Network',
        'description': 'The language network is a common fMRI network involved in comprehension and production of language. Core regions include inferior frontal gyrus, posterior superior temporal gyrus, middle temporal gyrus, angular gyrus, and supramarginal gyrus. It supports speech comprehension, lexical access, semantic interpretation, and controlled language production.'
    },
    {
        'title': '[NETWORK]Limbic Network',
        'description': 'The limbic network is a common fMRI network associated with affective and motivational processing. Core regions include amygdala, hippocampus, parahippocampal cortex, ventromedial prefrontal cortex, orbitofrontal cortex, and temporal pole. It supports emotional evaluation, memory-guided appraisal, affective meaning, and motivational relevance.'
    },
    {
        'title': '[NETWORK]Reward Network',
        'description': 'The reward network is a common fMRI network involved in valuation and reinforcement. Core regions include ventral striatum, nucleus accumbens, ventromedial prefrontal cortex, orbitofrontal cortex, anterior cingulate cortex, and midbrain. It supports reward anticipation, subjective value, reinforcement learning, motivation, and choice behavior.'
    },
    {
        'title': '[NETWORK]Memory Network',
        'description': 'The memory network is a common fMRI network involved in encoding and retrieving experiences. Core regions include hippocampus, parahippocampal cortex, entorhinal cortex, retrosplenial cortex, posterior cingulate cortex, and medial prefrontal cortex. It supports episodic memory, contextual representation, recollection, and scene-based mental construction.'
    },
    {
        'title': '[REGION]Medial Prefrontal Cortex',
        'description': 'The medial prefrontal cortex is a core cortical node commonly associated with internally directed and evaluative cognition. Anatomically, it includes medial portions of the superior frontal gyrus, rostral anterior cingulate cortex, and ventromedial prefrontal cortex. It supports self-referential evaluation, social inference, valuation, and internally generated thought.'
    },
    {
        'title': '[REGION]Posterior Cingulate Cortex',
        'description': 'The posterior cingulate cortex is a midline cortical region commonly involved in default mode processing. Anatomically, it lies in the posterior cingulate gyrus along the medial parietal cortex. It supports autobiographical memory, self-related thought, contextual integration, and internally oriented attention.'
    },
    {
        'title': '[REGION]Precuneus',
        'description': 'The precuneus is a medial parietal region commonly implicated in internally directed cognition. Anatomically, it lies anterior to the cuneus, posterior to the paracentral lobule, and superior to the posterior cingulate cortex. It supports visuospatial imagery, self-related processing, episodic retrieval, and mental scene construction.'
    },
    {
        'title': '[REGION]Angular Gyrus',
        'description': 'The angular gyrus is an inferior parietal region involved in semantic, social, and integrative cognition. Anatomically, it occupies the posterior inferior parietal lobule near the junction of parietal, temporal, and occipital cortex. It supports semantic integration, conceptual processing, number-related cognition, and aspects of internally guided thought.'
    },
    {
        'title': '[REGION]Hippocampus',
        'description': 'The hippocampus is a medial temporal lobe structure central to memory representation. Anatomically, it includes the hippocampal formation along the medial temporal lobe, including the dentate gyrus, CA fields, and subiculum. It supports episodic memory, relational binding, spatial context, and flexible retrieval of past experiences.'
    },
    {
        'title': '[REGION]Parahippocampal Cortex',
        'description': 'The parahippocampal cortex is a medial temporal cortical region associated with contextual and scene processing. Anatomically, it lies along the parahippocampal gyrus adjacent to the hippocampus. It supports contextual memory, scene representation, environmental layout processing, and memory-guided perception.'
    },
    {
        'title': '[REGION]Dorsolateral Prefrontal Cortex',
        'description': 'The dorsolateral prefrontal cortex is a lateral frontal region commonly involved in executive control. Anatomically, it includes middle frontal gyrus and adjacent superior frontal sulcus regions on the lateral prefrontal surface. It supports working memory, rule maintenance, planning, inhibition, and goal-directed cognition.'
    },
    {
        'title': '[REGION]Lateral Frontopolar Cortex',
        'description': 'The lateral frontopolar cortex is an anterior prefrontal region involved in high-level control. Anatomically, it occupies the lateral anterior frontal pole, primarily within rostral middle frontal and frontopolar cortex. It supports abstraction, prospective planning, relational reasoning, and coordination of multiple goals.'
    },
    {
        'title': '[REGION]Inferior Parietal Lobule',
        'description': 'The inferior parietal lobule is a lateral parietal region involved in attention and cognitive control. Anatomically, it includes the supramarginal gyrus and angular gyrus along the posterior lateral parietal cortex. It supports attentional selection, working memory, semantic integration, and flexible task representation.'
    },
    {
        'title': '[REGION]Intraparietal Sulcus',
        'description': 'The intraparietal sulcus is a dorsal parietal region involved in attention and spatial representation. Anatomically, it runs along the lateral parietal cortex between superior and inferior parietal regions. It supports visuospatial attention, numerical representation, sensorimotor transformation, and goal-directed selection.'
    },
    {
        'title': '[REGION]Anterior Insula',
        'description': 'The anterior insula is a frontal-insular region commonly associated with salience and interoception. Anatomically, it occupies the anterior portion of the insular cortex deep within the lateral sulcus. It supports awareness of bodily states, salience detection, affective experience, and cognitive state switching.'
    },
    {
        'title': '[REGION]Dorsal Anterior Cingulate Cortex',
        'description': 'The dorsal anterior cingulate cortex is a medial frontal region involved in control and performance monitoring. Anatomically, it lies on the dorsal bank of the anterior cingulate gyrus above the corpus callosum. It supports conflict monitoring, effort allocation, action selection, and adaptive cognitive control.'
    },
    {
        'title': '[REGION]Frontal Operculum',
        'description': 'The frontal operculum is an inferior frontal region commonly grouped with salience and task-control systems. Anatomically, it lies along the posterior inferior frontal cortex overlying the anterior insula. It supports salience processing, response selection, speech-related control, and task-set maintenance.'
    },
    {
        'title': '[REGION]Presupplementary Motor Area',
        'description': 'The presupplementary motor area is a medial frontal region involved in controlled action selection. Anatomically, it lies anterior to the supplementary motor area on the medial superior frontal gyrus. It supports response selection, sequence control, conflict processing, and preparation of internally guided actions.'
    },
    {
        'title': '[REGION]Frontal Eye Fields',
        'description': 'The frontal eye fields are dorsal frontal regions involved in eye movements and spatial attention. Anatomically, they lie near the intersection of the precentral sulcus and superior frontal sulcus. They support voluntary orienting, saccade preparation, visual search, and top-down spatial attention.'
    },
    {
        'title': '[REGION]Superior Parietal Lobule',
        'description': 'The superior parietal lobule is a dorsal parietal region involved in spatial attention and sensorimotor integration. Anatomically, it lies superior to the intraparietal sulcus on the lateral and medial parietal surface. It supports spatial orienting, visually guided action, body-space mapping, and attentional priority.'
    },
    {
        'title': '[REGION]Temporoparietal Junction',
        'description': 'The temporoparietal junction is a lateral cortical region involved in attentional reorienting and social cognition. Anatomically, it lies at the junction of posterior superior temporal cortex, inferior parietal lobule, and lateral occipital cortex. It supports detection of unexpected events, perspective-taking, attentional shifting, and socially relevant interpretation.'
    },
    {
        'title': '[REGION]Inferior Frontal Gyrus',
        'description': 'The inferior frontal gyrus is a lateral frontal region involved in language and cognitive control. Anatomically, it includes pars opercularis, pars triangularis, and pars orbitalis on the ventrolateral prefrontal cortex. It supports controlled retrieval, speech production, response inhibition, and selection among competing representations.'
    },
    {
        'title': '[REGION]Middle Frontal Gyrus',
        'description': 'The middle frontal gyrus is a lateral frontal region commonly involved in executive and attentional control. Anatomically, it lies between the superior frontal sulcus and inferior frontal sulcus on the lateral frontal lobe. It supports working memory, cognitive flexibility, attentional control, and goal maintenance.'
    },
    {
        'title': '[REGION]Precentral Gyrus',
        'description': 'The precentral gyrus is the primary motor cortical region. Anatomically, it lies immediately anterior to the central sulcus and contains the primary motor cortex. It supports voluntary movement execution, somatotopic motor control, and preparation of body-part-specific actions.'
    },
    {
        'title': '[REGION]Postcentral Gyrus',
        'description': 'The postcentral gyrus is the primary somatosensory cortical region. Anatomically, it lies immediately posterior to the central sulcus and contains primary somatosensory cortex. It supports tactile perception, proprioception, body-part localization, and sensory feedback for action.'
    },
    {
        'title': '[REGION]Supplementary Motor Area',
        'description': 'The supplementary motor area is a medial frontal motor region involved in action preparation. Anatomically, it lies on the medial superior frontal gyrus anterior to the primary motor leg area. It supports motor sequencing, internally generated movement, bimanual coordination, and action initiation.'
    },
    {
        'title': '[REGION]Primary Visual Cortex',
        'description': 'The primary visual cortex is the earliest cortical region for visual processing. Anatomically, it lies along the calcarine sulcus in medial occipital cortex. It supports retinotopic visual representation, edge detection, contrast processing, and basic spatial visual analysis.'
    },
    {
        'title': '[REGION]Extrastriate Visual Cortex',
        'description': 'Extrastriate visual cortex is a set of occipital regions involved in visual feature processing beyond primary visual cortex. Anatomically, it includes secondary and associative visual areas surrounding the calcarine cortex in lateral and ventral occipital cortex. It supports processing of shape, color, motion, object structure, and visual spatial relationships.'
    },
    {
        'title': '[REGION]Lateral Occipital Cortex',
        'description': 'The lateral occipital cortex is a visual association region involved in object-related perception. Anatomically, it lies on the lateral occipital surface posterior to temporal and parietal association cortex. It supports visual object recognition, shape analysis, and integration of visual contours into coherent forms.'
    },
    {
        'title': '[REGION]Fusiform Gyrus',
        'description': 'The fusiform gyrus is a ventral temporal-occipital region involved in high-level visual recognition. Anatomically, it lies on the ventral surface of the temporal and occipital lobes between the collateral and occipitotemporal sulci. It supports recognition of faces, words, objects, and visually learned categories.'
    },
    {
        'title': '[REGION]Middle Temporal Visual Area',
        'description': 'The middle temporal visual area is a lateral occipitotemporal region involved in motion perception. Anatomically, it lies near the posterior inferior temporal sulcus and lateral occipital-temporal cortex. It supports visual motion analysis, direction sensitivity, biological motion perception, and dynamic visual tracking.'
    },
    {
        'title': '[REGION]Heschl Gyrus',
        'description': 'Heschl gyrus is the primary auditory cortical region. Anatomically, it lies on the superior temporal plane within the lateral sulcus. It supports basic auditory perception, frequency analysis, temporal sound processing, and early cortical representation of acoustic input.'
    },
    {
        'title': '[REGION]Superior Temporal Gyrus',
        'description': 'The superior temporal gyrus is a lateral temporal region involved in auditory and language processing. Anatomically, it lies along the upper temporal lobe inferior to the lateral sulcus. It supports speech perception, auditory association, phonological analysis, and socially relevant sound processing.'
    },
    {
        'title': '[REGION]Superior Temporal Sulcus',
        'description': 'The superior temporal sulcus is a lateral temporal region involved in social and audiovisual perception. Anatomically, it lies between the superior temporal gyrus and middle temporal gyrus. It supports biological motion perception, voice processing, audiovisual integration, and interpretation of socially meaningful cues.'
    },
    {
        'title': '[REGION]Middle Temporal Gyrus',
        'description': 'The middle temporal gyrus is a lateral temporal region involved in semantic and language processing. Anatomically, it lies between the superior temporal sulcus and inferior temporal sulcus. It supports lexical-semantic access, conceptual representation, sentence comprehension, and meaning-based language processing.'
    },
    {
        'title': '[REGION]Supramarginal Gyrus',
        'description': 'The supramarginal gyrus is an inferior parietal region involved in phonological and sensorimotor integration. Anatomically, it curves around the posterior end of the lateral sulcus within the inferior parietal lobule. It supports phonological processing, speech sound mapping, action representation, and attentional selection.'
    },
    {
        'title': '[REGION]Amygdala',
        'description': 'The amygdala is a medial temporal lobe structure involved in affective salience. Anatomically, it lies anterior to the hippocampus within the medial temporal lobe. It supports emotional appraisal, threat relevance, affective learning, and detection of motivationally significant stimuli.'
    },
    {
        'title': '[REGION]Ventromedial Prefrontal Cortex',
        'description': 'The ventromedial prefrontal cortex is a medial frontal region involved in valuation and affective meaning. Anatomically, it lies along the ventral medial surface of the prefrontal cortex, including medial orbitofrontal and subgenual prefrontal regions. It supports subjective value, emotion-guided decision-making, self-relevance, and integration of affective information.'
    },
    {
        'title': '[REGION]Orbitofrontal Cortex',
        'description': 'The orbitofrontal cortex is a ventral frontal region involved in valuation and outcome representation. Anatomically, it lies on the orbital surface of the frontal lobe above the eye sockets. It supports reward valuation, sensory-affective integration, outcome prediction, and flexible updating of preferences.'
    },
    {
        'title': '[REGION]Ventral Striatum',
        'description': 'The ventral striatum is a subcortical region involved in reward and motivation. Anatomically, it includes nucleus accumbens and adjacent ventral portions of caudate and putamen. It supports reward anticipation, motivational drive, reinforcement learning, and action value representation.'
    },
    {
        'title': '[REGION]Caudate',
        'description': 'The caudate is a dorsal striatal structure involved in goal-directed action and cognitive control. Anatomically, it includes the head, body, and tail of the caudate nucleus along the lateral ventricle. It supports action selection, learning from feedback, procedural control, and flexible updating of behavior.'
    },
    {
        'title': '[REGION]Putamen',
        'description': 'The putamen is a dorsal striatal structure involved in motor and habit-related processing. Anatomically, it lies lateral to the globus pallidus within the basal ganglia. It supports movement regulation, action sequencing, reinforcement-based learning, and sensorimotor habit formation.'
    },
    {
        'title': '[REGION]Thalamus',
        'description': 'The thalamus is a subcortical relay and integration structure involved in sensory and cognitive processing. Anatomically, it lies bilaterally in the dorsal diencephalon on either side of the third ventricle. It supports sensory relay, attentional gating, motor coordination, arousal, and cortico-subcortical communication.'
    },
    {
        'title': '[REGION]Cerebellum',
        'description': 'The cerebellum is a posterior brain structure involved in motor coordination and cognitive timing. Anatomically, it lies beneath the occipital and temporal lobes in the posterior fossa and includes cerebellar hemispheres and vermis. It supports coordination, error correction, timing, motor learning, and aspects of cognitive prediction.'
    },
    {
        'title': '[FUNCTION]Self-Referential Processing',
        'description': 'Self-referential processing is the cognitive process of evaluating information in relation to oneself. Core regions include medial prefrontal cortex, posterior cingulate cortex, precuneus, and angular gyrus. It supports reflection on personal traits, preferences, autobiographical relevance, and internally oriented evaluation.'
    },
    {
        'title': '[FUNCTION]Autobiographical Memory',
        'description': 'Autobiographical memory is the cognitive process of retrieving personally experienced events. Core regions include hippocampus, posterior cingulate cortex, medial prefrontal cortex, precuneus, and angular gyrus. It supports recollection of past experiences, contextual detail, personal narrative, and memory-based self-continuity.'
    },
    {
        'title': '[FUNCTION]Episodic Memory',
        'description': 'Episodic memory is the cognitive process of encoding and retrieving specific events. Core regions include hippocampus, parahippocampal cortex, entorhinal cortex, posterior cingulate cortex, and medial prefrontal cortex. It supports relational binding, contextual recall, temporal organization, and reconstruction of past experiences.'
    },
    {
        'title': '[FUNCTION]Working Memory',
        'description': 'Working memory is the cognitive process of maintaining and manipulating information over short intervals. Core regions include dorsolateral prefrontal cortex, inferior parietal lobule, intraparietal sulcus, and lateral frontopolar cortex. It supports active maintenance, updating, manipulation, and goal-relevant use of information.'
    },
    {
        'title': '[FUNCTION]Executive Control',
        'description': 'Executive control is the cognitive process of regulating thought and action according to goals. Core regions include dorsolateral prefrontal cortex, lateral frontopolar cortex, inferior parietal lobule, dorsal anterior cingulate cortex, and presupplementary motor area. It supports planning, inhibition, task switching, rule maintenance, and adaptive behavior.'
    },
    {
        'title': '[FUNCTION]Cognitive Flexibility',
        'description': 'Cognitive flexibility is the ability to shift between rules, goals, or mental representations. Core regions include dorsolateral prefrontal cortex, lateral frontopolar cortex, inferior parietal lobule, anterior insula, and dorsal anterior cingulate cortex. It supports task switching, strategy updating, adaptive control, and flexible selection of behavior.'
    },
    {
        'title': '[FUNCTION]Salience Detection',
        'description': 'Salience detection is the cognitive process of identifying information that is behaviorally relevant. Core regions include anterior insula, dorsal anterior cingulate cortex, frontal operculum, amygdala, and ventral striatum. It supports prioritization of important stimuli, detection of internal bodily signals, and allocation of cognitive resources.'
    },
    {
        'title': '[FUNCTION]Interoception',
        'description': 'Interoception is the cognitive process of sensing and interpreting internal bodily states. Core regions include anterior insula, posterior insula, dorsal anterior cingulate cortex, and somatosensory cortex. It supports awareness of bodily signals, affective feeling states, physiological regulation, and subjective internal experience.'
    },
    {
        'title': '[FUNCTION]Conflict Monitoring',
        'description': 'Conflict monitoring is the cognitive process of detecting competition between actions or representations. Core regions include dorsal anterior cingulate cortex, presupplementary motor area, anterior insula, and dorsolateral prefrontal cortex. It supports performance monitoring, control adjustment, response selection, and reduction of cognitive interference.'
    },
    {
        'title': '[FUNCTION]Spatial Attention',
        'description': 'Spatial attention is the cognitive process of selectively prioritizing locations in visual space. Core regions include frontal eye fields, intraparietal sulcus, superior parietal lobule, and dorsal occipital cortex. It supports visual search, spatial orienting, attentional selection, and preparation for perception or action.'
    },
    {
        'title': '[FUNCTION]Attentional Reorienting',
        'description': 'Attentional reorienting is the cognitive process of shifting attention toward unexpected relevant events. Core regions include temporoparietal junction, inferior frontal gyrus, middle frontal gyrus, and ventral frontal cortex. It supports detection of behaviorally important changes, interruption of ongoing focus, and redirection of attentional priority.'
    },
    {
        'title': '[FUNCTION]Visual Perception',
        'description': 'Visual perception is the cognitive process of interpreting information from the eyes. Core regions include primary visual cortex, extrastriate visual cortex, lateral occipital cortex, fusiform gyrus, and middle temporal visual area. It supports detection of shape, color, motion, object identity, and spatial visual structure.'
    },
    {
        'title': '[FUNCTION]Object Recognition',
        'description': 'Object recognition is the cognitive process of identifying visual forms as meaningful objects. Core regions include lateral occipital cortex, fusiform gyrus, inferior temporal cortex, and extrastriate visual cortex. It supports category recognition, form analysis, visual familiarity, and mapping of visual input to semantic identity.'
    },
    {
        'title': '[FUNCTION]Motion Perception',
        'description': 'Motion perception is the cognitive process of detecting and interpreting movement in visual input. Core regions include middle temporal visual area, extrastriate visual cortex, lateral occipital cortex, and superior temporal sulcus. It supports tracking movement direction, speed, biological motion, and dynamic visual scenes.'
    },
    {
        'title': '[FUNCTION]Auditory Perception',
        'description': 'Auditory perception is the cognitive process of interpreting sound. Core regions include Heschl gyrus, planum temporale, superior temporal gyrus, and superior temporal sulcus. It supports acoustic feature analysis, sound localization, speech sound perception, and interpretation of meaningful auditory events.'
    },
    {
        'title': '[FUNCTION]Language Comprehension',
        'description': 'Language comprehension is the cognitive process of extracting meaning from spoken or written language. Core regions include posterior superior temporal gyrus, middle temporal gyrus, angular gyrus, supramarginal gyrus, and inferior frontal gyrus. It supports word meaning, sentence interpretation, phonological analysis, and integration of linguistic context.'
    },
    {
        'title': '[FUNCTION]Speech Production',
        'description': 'Speech production is the cognitive process of generating articulated language. Core regions include inferior frontal gyrus, premotor cortex, supplementary motor area, precentral gyrus, and superior temporal gyrus. It supports lexical selection, phonological planning, articulatory preparation, and fluent verbal output.'
    },
    {
        'title': '[FUNCTION]Semantic Processing',
        'description': 'Semantic processing is the cognitive process of representing and retrieving meaning. Core regions include middle temporal gyrus, angular gyrus, inferior frontal gyrus, anterior temporal cortex, and posterior superior temporal cortex. It supports conceptual knowledge, word meaning, category structure, and context-based interpretation.'
    },
    {
        'title': '[FUNCTION]Motor Execution',
        'description': 'Motor execution is the cognitive-motor process of producing voluntary movement. Core regions include precentral gyrus, supplementary motor area, putamen, thalamus, and cerebellum. It supports body-part-specific movement, action timing, motor coordination, and implementation of intended actions.'
    },
    {
        'title': '[FUNCTION]Motor Planning',
        'description': 'Motor planning is the cognitive-motor process of preparing actions before execution. Core regions include premotor cortex, supplementary motor area, presupplementary motor area, posterior parietal cortex, and cerebellum. It supports action sequencing, response preparation, movement selection, and transformation of goals into motor commands.'
    },
    {
        'title': '[FUNCTION]Somatosensory Processing',
        'description': 'Somatosensory processing is the cognitive process of interpreting bodily sensation. Core regions include postcentral gyrus, central operculum, posterior insula, and secondary somatosensory cortex. It supports touch, proprioception, body localization, pain-related sensation, and sensory feedback for movement.'
    },
    {
        'title': '[FUNCTION]Emotion Processing',
        'description': 'Emotion processing is the cognitive-affective process of evaluating affective significance. Core regions include amygdala, anterior insula, ventromedial prefrontal cortex, orbitofrontal cortex, and dorsal anterior cingulate cortex. It supports affective appraisal, emotional awareness, motivational relevance, and regulation of emotional responses.'
    },
    {
        'title': '[FUNCTION]Reward Valuation',
        'description': 'Reward valuation is the cognitive process of assigning subjective value to outcomes or options. Core regions include ventral striatum, nucleus accumbens, ventromedial prefrontal cortex, orbitofrontal cortex, and anterior cingulate cortex. It supports reward anticipation, preference formation, motivational drive, and value-guided choice.'
    },
    {
        'title': '[FUNCTION]Decision Making',
        'description': 'Decision making is the cognitive process of selecting among possible actions or options. Core regions include ventromedial prefrontal cortex, dorsolateral prefrontal cortex, orbitofrontal cortex, anterior cingulate cortex, and ventral striatum. It supports comparison of options, integration of value and goals, uncertainty evaluation, and action selection.'
    },
    {
        'title': '[FUNCTION]Social Cognition',
        'description': 'Social cognition is the cognitive process of interpreting people, intentions, and social meaning. Core regions include medial prefrontal cortex, temporoparietal junction, posterior cingulate cortex, superior temporal sulcus, and temporal pole. It supports perspective-taking, trait inference, social prediction, and interpretation of socially relevant information.'
    },
])
df_more = pd.DataFrame((df_more["title"] + "\n" + df_more["description"]).tolist(), columns=["text"])
df_pubs = pd.concat(
    (
        df_pubs[["text", "title_canonical"]],
        df_func,
        df_func_2,
        df_more
    ),
    ignore_index=True,
).reset_index(drop=True)

df_pubs.reset_index(inplace=True, drop=True)
del df_pubs['title_canonical']
df_pubs.shape, df_pubs["text"].str.split("]").str[0].value_counts()

# %%
# df_pubs = df_pubs[~df_pubs["title_canonical"].isna()]
df_pubs.reset_index(inplace=True, drop=True)

terms_to_drop = [
    "Theory of Mind Network",
    "Theory of Mind",
    "Social Cognition Network",
    #"Social Cognition",
    "Empathy Network",
    #"Empathy",
    "Biological Motion",
    #"Action Observation",

    "Emotion Network",
    #"Pain Network",
    #"Pain Processing",
    "Affective Network",
    "Orbitofrontal-Affective Network",
    "Arousal Network",
    #"Arousal",
    "Autonomic Regulation",
    "Conscious Awareness",

    "Music Processing",
    "Numerical Cognition",
    "Numerical Cognition Network",
    "Calculation",
    "Reading Network",
    "Reading",
    "Visual Word Form Processing",
    "Visual Word Form Area",

    "Speech Production Network",
    "Speech Perception Network",
    #"Speech Production",
    #"Speech Perception",
    "Phonological Processing",
    "Syntax Processing",
    "Lexical Retrieval",

    "Object Recognition",
    "Object Perception Network",
    "Face Processing",
    "Face Perception Network",
    "Scene Perception",
    "Scene Perception Network",
    "Body Perception",
    "Body Representation",
    "Voice Perception",
    "Voice-Selective Auditory Cortex",

    "Motion Perception",
    "Color Perception",
    "Motion-Selective Visual Cortex",
    "Color-Selective Visual Cortex",

    "Olfaction",
    "Olfactory Processing",
    "Olfactory Cortex",
    "Gustation",
    "Gustatory Processing",
    "Primary Gustatory Cortex",
    "Vestibular Processing",
    "Vestibular Cortex",

    #"Motor Learning",
    "Procedural Learning",
    "Hand Movement",
    "Eye Movements",
    "Oculomotor Control",
    "Oculomotor Network",
    "Hand Motor Cortex",
    "Foot Motor Cortex",
    "Face Motor Cortex",
    "Hand Sensorimotor Network",
    "Foot Sensorimotor Network",
    "Mouth Sensorimotor Network",

    "Verbal Working Memory",
    "Spatial Working Memory",
    "Cognitive Flexibility",
    "Conflict Monitoring",
    "Error Monitoring",
    #"Decision Making",
    "Value Representation",
    "Reinforcement Learning",
    "Habit Learning",

    "Emotion Processing",
    "Fear Processing",
    "Emotion Regulation",

    #"Hippocampal Memory",
    #"Hippocampal Function",
    #"Amygdala Function",
    #"Insula Function",
    #"Thalamic Function",
    #"Cerebellar Function",

    #"Episodic Memory",
    #"Memory Encoding",
    #"Memory Retrieval",
    "Semantic Memory",
    #"Autobiographical Memory",
    "Prospective Thinking",
    "Mental Imagery",

    #"Spatial Navigation",
    "Spatial Navigation Network",
    #"Navigation",
    "Retrosplenial Cortex",
    "Parahippocampal Place Area",

    "Medial Temporal Memory Network",
    "Hippocampal Network",
    "Posterior Medial Network",

    #"Subcortical Network",
    "Basal Ganglia Network",
    "Fronto-Striatal Network",
    "Thalamic Network",
    "Cerebellar Network",
    "Cerebro-Cerebellar Network",

    "Primary Visual Network",
    "Visual Association Network",
    "Primary Auditory Network",
    "Auditory Association Network",
    "Dorsal Visual Stream",
    "Ventral Visual Stream",
    "Dorsal Language Stream",
    "Ventral Language Stream",

    #"Multiple Demand Network",
    "Salience/Ventral Attention Network",
    "Interoceptive Network",
    "Visuospatial Network",
]

# df_pubs = df_pubs[~df_pubs["title_canonical"].isin(terms_to_drop)].reset_index(drop=True)
# df_pubs.reset_index(inplace=True, drop=True)


df_pubs["title"] = df_pubs['text'].str.split("]").str[1].str.split("\n").str[0]
t = (df_pubs["title"] + " [SEP] " + df_pubs["text"].str.replace(".*\n", "", regex=True))

t = t.tolist()
# df_pubs['text'] = df_pubs["text"].str.split("]").str[0] + "]" + df_pubs["title"] + "\n" + df_pubs["text"].str.replace(".*\n", "", regex=True)
df_pubs['text'] = df_pubs["text"].str.split("]").str[0] + "]" + df_pubs["title"] + "\n" + df_pubs["text"].str.replace(".*\n", "", regex=True)


# %%
df_pubs.shape, df_pubs["text"].str.split("]").str[0].value_counts()
# ((1261, 3),
#  text
#  [REGION      754
#  [FUNCTION    306
#  [NETWORK     201
#  Name: count, dtype: int64)

# %%
# load models
specter = load_model("specter")
specter = specter.to("cuda")
specter.specter = specter.specter.eval()
masker = load_masker()
adapter = torch.load(ADAPTER_PATH, weights_only=False).cuda().eval()

ae = load_model("autoencoder")
ae = ae.cpu()
enc = ae.encoder.cuda().eval()

DEVICE = "cuda"
proj_head_image = load_model("proj_head_image_infonce").to(DEVICE).eval()
proj_head_text = load_model("proj_head_text_infonce").to(DEVICE).eval()

batch_size = 2048

latent_images = torch.zeros((len(t), 384))
latent_text = torch.zeros((len(t), 768))
for i in tqdm(range(0, len(df_pubs), batch_size), total=len(df_pubs)//batch_size):
    with torch.no_grad():
        _text = F.normalize(specter(t[i:i+batch_size]), dim=1)
        latent_text[i:i+len(_text)] = _text
        _img = torch.sigmoid(adapter(_text)).float()
        latent_images[i:i+len(_img)] = enc(_img)
latent_images = latent_images.to("cuda")
latent_text = latent_text.to("cuda")


# %%
@torch.no_grad()
def project_images_batched(x, batch_size=1024):
    outs = []
    for start in tqdm(range(0, len(x), batch_size), desc="project image semantics"):
        batch = x[start:start + batch_size].to(DEVICE, non_blocking=True)
        outs.append(proj_head_image(batch).detach().float().cpu())
    return F.normalize(torch.cat(outs, dim=0), dim=1)

@torch.no_grad()
def project_text_batched(x, batch_size=1024):
    outs = []
    for start in tqdm(range(0, len(x), batch_size), desc="project text semantics"):
        batch = x[start:start + batch_size].to(DEVICE, non_blocking=True)
        outs.append(proj_head_text(batch).detach().float().cpu())
    return F.normalize(torch.cat(outs, dim=0), dim=1)

image_semantic = project_images_batched(latent_images)
text_semantic = project_text_batched(latent_text)

cs = image_semantic @ text_semantic.T

k = 50
titles = df_pubs["title"].to_numpy()
s_topk = cs.argsort(dim=1, descending=True).numpy()[:, :k]
m = (titles[s_topk] == titles[:, None]).any(axis=1)

df_pubs = df_pubs[m]
df_pubs.reset_index(inplace=True, drop=True)

latent_images = latent_images[torch.from_numpy(m)]
latent_text = latent_text[torch.from_numpy(m)]
image_semantic = image_semantic[torch.from_numpy(m)]
text_semantic = text_semantic[torch.from_numpy(m)]

# %%
df_pubs.shape, df_pubs["text"].str.split("]").str[0].value_counts()

# %%
df_ref = df_pubs.copy()

CANONICAL_PROJ_TEMP = 0.05
SEED = 123
seed_everything(SEED)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
latent_images = torch.as_tensor(latent_images, dtype=torch.float32).cpu().clone()
if len(latent_images) != len(df_pubs):
    raise RuntimeError(f"Training alignment mismatch: latent_images={len(latent_images)} df_pubs={len(df_pubs)}")
if len(latent_text) != len(df_pubs):
    raise RuntimeError(f"Training alignment mismatch: latent_text={len(latent_text)} df_pubs={len(df_pubs)}")

device = "cuda"

qformer = QFormer(
    image_dim=384,
    semantic_dim=384,
    lm_dim=1024,
    num_queries=32,
    hidden_dim=512,
    num_heads=8,
    num_layers=6,
).to(device).to(torch.bfloat16)

model_path = GROUNDED_QFORMER_PATH
qformer.load_state_dict(torch.load(model_path, map_location=device))

qformer = qformer.to(device).to(train_dtype)
qformer.train()
print(f"Rows: {len(df_pubs):,} | device: {device} | qformer dtype: {next(qformer.parameters()).dtype}")


# %%
df_ref = df_pubs.copy()
canonical_semantic_bank = F.normalize(text_semantic.float(), dim=1).cpu()
canonical_concepts = (
    df_ref["text"].astype(str).str.extract(r"^\[([A-Z]+)\]", expand=False)
    .str.lower()
    .fillna("unknown")
    .reset_index(drop=True)
)
canonical_semantic_banks = {"all": canonical_semantic_bank}
for _basis_name in ("network", "region", "function"):
    _idx = np.where(canonical_concepts.to_numpy() == _basis_name)[0]
    if len(_idx):
        canonical_semantic_banks[_basis_name] = canonical_semantic_bank[_idx]

E = canonical_semantic_bank

def get_canonical_bank(canonical_banks, basis):
    basis = str(basis).lower()
    basis = {"networks": "network", "regions": "region", "functions": "function"}.get(basis, basis)
    if basis not in canonical_banks:
        valid = ", ".join(sorted(canonical_banks))
        raise ValueError(f"Unknown canonical basis {basis!r}; valid options: {valid}")
    return canonical_banks[basis]

@torch.no_grad()
def project_to_canonical(E, x, *, temp, return_weights=False):
    x = torch.as_tensor(x, dtype=torch.float32)
    x = x.reshape(1, -1) if x.ndim == 1 else x
    E = F.normalize(E.to(device=x.device, dtype=torch.float32), dim=1)
    x = F.normalize(x.float(), dim=1)
    logits = x @ E.T
    w = torch.softmax(logits / temp, dim=-1)
    z = F.normalize(w @ E, dim=1)
    if return_weights:
        return z, w
    return z

train_semantic_images = project_to_canonical(canonical_semantic_bank, image_semantic, temp=CANONICAL_PROJ_TEMP).cpu()
train_semantic_text = project_to_canonical(canonical_semantic_bank, text_semantic, temp=CANONICAL_PROJ_TEMP).cpu()

print("image_semantic:", tuple(image_semantic.shape))
print("text_semantic:", tuple(text_semantic.shape))
print("canonical_semantic_bank:", tuple(canonical_semantic_bank.shape))
print("canonical basis sizes:", {k: tuple(v.shape) for k, v in canonical_semantic_banks.items()})
print("train_semantic_images:", tuple(train_semantic_images.shape))
print("train_semantic_text:", tuple(train_semantic_text.shape))


# %%
# Load the frozen LM that matches the 1024-dim permissive Q-Former checkpoint.
LM_MODEL = NEURO_QWEN_REPO_ID
model, tokenizer = load_huggingface_model(LM_MODEL, device=str(device), dtype=train_dtype)
for p in model.parameters():
    p.requires_grad = False
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.eval()  # Frozen LM: keep dropout disabled.

lm_dim = model.config.hidden_size
assert lm_dim == 1024, f"Expected lm_dim=1024 for loaded qformer checkpoint, got {lm_dim}"
PAD_ID = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
EOS_ID = tokenizer.eos_token_id
model_dtype = next(model.parameters()).dtype

with torch.no_grad():
    LM_EMB_NORM = model.get_input_embeddings().weight.detach().float().norm(dim=1).mean().item()
print(f"LM hidden size: {lm_dim} | dtype: {model_dtype} | mean token norm: {LM_EMB_NORM:.2f}")


# %%
MAX_TOKENS = 256
TRAIN_BATCH = 18
EVAL_BATCH = 18
NUM_WORKERS = 0
VAL_FRAC = 0.1

# %%
RAW_IMAGE_INPUT_MODE = "raw"  # "projected", "raw", or "zero"
if RAW_IMAGE_INPUT_MODE not in {"projected", "raw", "zero"}:
    raise ValueError(f"RAW_IMAGE_INPUT_MODE must be 'projected', 'raw', or 'zero', got {RAW_IMAGE_INPUT_MODE!r}")
USE_RAW_IMAGE_INPUT = RAW_IMAGE_INPUT_MODE != "zero"

text_arr = np.array(df_pubs["text"].astype(str).values, dtype=object)
train_raw_images = latent_images if RAW_IMAGE_INPUT_MODE == "raw" else train_semantic_images
train_semantic_text_for_loader = train_semantic_text
train_semantic_image_for_loader = train_semantic_images

print("raw input:", RAW_IMAGE_INPUT_MODE)
print("semantic input: mixed canonical text/image projection during training; canonical image projection at eval")

def clean_space(x):
    return re.sub(r"\s+", " ", str(x).replace("\n", " ").strip())

class BrainGroundedTextDataset(Dataset):
    def __init__(self, raw_images, semantic_text_images, semantic_image_images, texts, tokenizer, max_tokens=MAX_TOKENS):
        if not (len(raw_images) == len(semantic_text_images) == len(semantic_image_images) == len(texts)):
            raise RuntimeError(
                f"Dataset alignment mismatch: raw_images={len(raw_images)} "
                f"semantic_text_images={len(semantic_text_images)} "
                f"semantic_image_images={len(semantic_image_images)} texts={len(texts)}"
            )
        self.raw_images = torch.as_tensor(raw_images, dtype=torch.float32).cpu().clone()
        self.semantic_text_images = torch.as_tensor(semantic_text_images, dtype=torch.float32).cpu().clone()
        self.semantic_image_images = torch.as_tensor(semantic_image_images, dtype=torch.float32).cpu().clone()
        self.texts = np.asarray(texts, dtype=object)
        self.titles = np.array([str(x).split("\n", 1)[0] for x in self.texts], dtype=object)
        self.input_ids = []
        for text in self.texts:
            ids = tokenizer(str(text), truncation=True, max_length=max_tokens - 1, add_special_tokens=False)["input_ids"]
            if len(ids) == 0 or ids[-1] != EOS_ID:
                ids.append(EOS_ID)
            self.input_ids.append(torch.tensor(ids[:max_tokens], dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.raw_images[idx], self.semantic_text_images[idx], self.semantic_image_images[idx], self.input_ids[idx]

def collate_fn(batch):
    raw, sem_text, sem_image, input_ids = zip(*batch)
    raw = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in raw])
    sem_text = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in sem_text])
    sem_image = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in sem_image])
    lengths = torch.tensor([len(x) for x in input_ids], dtype=torch.long)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=PAD_ID)
    attn_mask = (torch.arange(input_ids.size(1))[None, :] < lengths[:, None]).long()
    return raw, sem_text, sem_image, input_ids, attn_mask


def make_train_val_loaders(raw_images, semantic_text_images, semantic_image_images, texts, tokenizer, *, val_frac=VAL_FRAC, seed=SEED, label="dataset"):
    raw_images = torch.as_tensor(raw_images, dtype=torch.float32).cpu()
    semantic_text_images = torch.as_tensor(semantic_text_images, dtype=torch.float32).cpu()
    semantic_image_images = torch.as_tensor(semantic_image_images, dtype=torch.float32).cpu()
    texts = np.asarray(texts, dtype=object)
    split_titles = np.array([str(x).split("\n", 1)[0] for x in texts], dtype=object)
    rng = np.random.default_rng(seed)

    train_idx = []
    val_idx = []
    df_split = pd.DataFrame({"idx": np.arange(len(texts), dtype=int), "title": split_titles})
    for _, group in df_split.groupby("title", sort=False, dropna=False):
        idx = group["idx"].to_numpy(dtype=int).copy()
        rng.shuffle(idx)
        if len(idx) <= 1:
            train_idx.extend(idx.tolist())
            continue
        n_val = max(1, int(round(len(idx) * val_frac)))
        n_val = min(n_val, len(idx) - 1)
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())

    train_idx = np.asarray(train_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)
    if len(val_idx) == 0:
        all_idx = np.arange(len(texts), dtype=int)
        rng.shuffle(all_idx)
        n_val = max(1, int(round(len(texts) * val_frac)))
        n_val = min(n_val, len(texts) - 1)
        val_idx = all_idx[:n_val]
        train_idx = all_idx[n_val:]

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    train_ds = BrainGroundedTextDataset(
        raw_images[train_idx],
        semantic_text_images[train_idx],
        semantic_image_images[train_idx],
        texts[train_idx],
        tokenizer,
    )
    val_ds = BrainGroundedTextDataset(
        raw_images[val_idx],
        semantic_text_images[val_idx],
        semantic_image_images[val_idx],
        texts[val_idx],
        tokenizer,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_BATCH,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        worker_init_fn=seed_worker,
        generator=make_torch_generator(seed + 101),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=EVAL_BATCH,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        worker_init_fn=seed_worker,
        generator=make_torch_generator(seed + 202),
    )
    print(f"{label}: train {len(train_ds):,} | val {len(val_ds):,} | train batches {len(train_loader):,} | val batches {len(val_loader):,}")
    print(f"{label}: title classes {pd.Series(split_titles).nunique():,}")
    print("Train sample:", tokenizer.decode(train_ds[0][-1], skip_special_tokens=True))
    print("Val sample:", tokenizer.decode(val_ds[0][-1], skip_special_tokens=True))
    return train_ds, val_ds, train_loader, val_loader, train_idx, val_idx

train_ds, val_ds, train_loader, val_loader, train_idx_clean, val_idx_clean = make_train_val_loaders(
    train_raw_images,
    train_semantic_text_for_loader,
    train_semantic_image_for_loader,
    text_arr,
    tokenizer,
    label="filtered network dataset",
)


# %%
USE_TITLE_LOSS_WEIGHTS = False
TITLE_LOSS_WEIGHT = None
BODY_LOSS_WEIGHT = None
TITLE_LOSS_FALLBACK_TOKENS = None
NEWLINE_TOKEN_IDS = tokenizer("\n", add_special_tokens=False)["input_ids"]
SEMANTIC_IMAGE_MIX_P = 0.5
SEMANTIC_VAL_USES_IMAGE = True

def apply_input_policy(raw_images, semantic_images):
    if not USE_RAW_IMAGE_INPUT:
        raw_images = torch.zeros_like(raw_images)
    return raw_images, semantic_images

def match_lm_token_norm(vis):
    target_norm = torch.tensor(LM_EMB_NORM, device=vis.device, dtype=vis.dtype)
    vis_norm = vis.float().norm(dim=-1, keepdim=True).clamp_min(1e-6).to(vis.dtype)
    return vis * (target_norm / vis_norm)

def title_loss_weights(input_ids, attn_mask):
    if not USE_TITLE_LOSS_WEIGHTS:
        return attn_mask.float()
    weights = torch.full(input_ids.shape, BODY_LOSS_WEIGHT, dtype=torch.float32, device=input_ids.device)
    title_mask = torch.zeros(input_ids.shape, dtype=torch.bool, device=input_ids.device)
    if len(NEWLINE_TOKEN_IDS) == 1:
        newline_id = NEWLINE_TOKEN_IDS[0]
        for i in range(input_ids.size(0)):
            hits = (input_ids[i] == newline_id).nonzero(as_tuple=False).flatten()
            end = int(hits[0].item()) + 1 if len(hits) else min(TITLE_LOSS_FALLBACK_TOKENS, input_ids.size(1))
            title_mask[i, :end] = True
    else:
        title_mask[:, :min(TITLE_LOSS_FALLBACK_TOKENS, input_ids.size(1))] = True
    return torch.where(title_mask, torch.as_tensor(TITLE_LOSS_WEIGHT, device=input_ids.device), weights) * attn_mask.float()

def select_semantic_input(semantic_text_images, semantic_image_images, *, training):
    if training:
        if SEMANTIC_IMAGE_MIX_P <= 0:
            return semantic_text_images
        if SEMANTIC_IMAGE_MIX_P >= 1:
            return semantic_image_images
        use_image = torch.rand(
            semantic_text_images.size(0),
            1,
            device=semantic_text_images.device,
        ) < SEMANTIC_IMAGE_MIX_P
        return torch.where(use_image, semantic_image_images, semantic_text_images)
    return semantic_image_images if SEMANTIC_VAL_USES_IMAGE else semantic_text_images
    # return semantic_image_images

def lm_loss(qformer, model, raw_images, semantic_text_images, semantic_image_images, input_ids, attn_mask, device):
    raw_images = raw_images.to(device, non_blocking=True)
    semantic_text_images = semantic_text_images.to(device, non_blocking=True)
    semantic_image_images = semantic_image_images.to(device, non_blocking=True)
    input_ids = input_ids.to(device, non_blocking=True)
    attn_mask = attn_mask.to(device, non_blocking=True)
    semantic_images = select_semantic_input(
        semantic_text_images,
        semantic_image_images,
        training=qformer.training,
    )
    raw_images, semantic_images = apply_input_policy(raw_images, semantic_images)
    batch_size = raw_images.size(0)

    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
        vis_tokens = qformer(raw_images.to(train_dtype), semantic_images.to(train_dtype))
        vis_tokens = match_lm_token_norm(vis_tokens).to(model_dtype)
        num_q = vis_tokens.size(1)
        with torch.no_grad():
            text_embeds = model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([vis_tokens, text_embeds.to(model_dtype)], dim=1)
        vis_mask = torch.ones(batch_size, num_q, device=device, dtype=torch.long)
        full_mask = torch.cat([vis_mask, attn_mask], dim=1)
        vis_labels = torch.full((batch_size, num_q), -100, dtype=torch.long, device=device)
        text_labels = input_ids.masked_fill(attn_mask == 0, -100)
        labels = torch.cat([vis_labels, text_labels], dim=1)
        text_weights = title_loss_weights(input_ids, attn_mask)
        vis_weights = torch.ones(batch_size, num_q, device=device, dtype=torch.float32)
        loss_weights = torch.cat([vis_weights, text_weights], dim=1)
        logits = model(inputs_embeds=inputs_embeds, attention_mask=full_mask, use_cache=False).logits
        shift_logits = logits[:, :-1].float().contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_weights = loss_weights[:, 1:].contiguous()
        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(shift_labels)
        valid = shift_labels.ne(-100)
        return (token_loss * shift_weights * valid).sum() / (shift_weights * valid).sum().clamp_min(1.0)

@torch.no_grad()
def mean_lm_loss(loader):
    qformer.eval()
    model.eval()
    total = 0.0
    for raw, sem_text, sem_image, input_ids, attn_mask in loader:
        total += lm_loss(qformer, model, raw, sem_text, sem_image, input_ids, attn_mask, device).item()
    return total / max(len(loader), 1)



# %%
seed_everything(SEED)


NUM_EPOCHS = 30
CHECKPOINT_PREFIX = CANONICAL_CHECKPOINT_PREFIX

optimizer = torch.optim.AdamW(qformer.parameters(), lr=1e-4, weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
history = {"train": [], "val": []}

for epoch in tqdm(range(NUM_EPOCHS), total=NUM_EPOCHS):
    qformer.train()
    model.eval()
    train_loss = 0.0
    for raw, sem_text, sem_image, input_ids, attn_mask in train_loader:
        optimizer.zero_grad(set_to_none=True)
        loss = lm_loss(qformer, model, raw, sem_text, sem_image, input_ids, attn_mask, device)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(qformer.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= max(len(train_loader), 1)
    val_loss = mean_lm_loss(val_loader) if val_loader is not None else float("nan")
    scheduler.step()
    history["train"].append(train_loss)
    history["val"].append(val_loss)
    print(f"Epoch {epoch + 1:3d} | train {train_loss:.4f} | val {val_loss:.4f} | lr {scheduler.get_last_lr()}")

torch.save(qformer.state_dict(), f"{CHECKPOINT_PREFIX}_epoch-{epoch}.pt")


# %%
import matplotlib.pyplot as plt

if history["train"]:
    plt.figure(figsize=(6, 4))
    plt.plot(history["train"], label="train")
    if history.get("val") and len(history["val"]) == len(history["train"]):
        plt.plot(history["val"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("LM loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


# %%
model_path = CANONICAL_QFORMER_PATH
qformer.load_state_dict(torch.load(model_path, map_location=device))

# %%
from neurovlm.data import load_masker, data_dir
from nilearn.image import resample_to_img
import nibabel as nib

def flatten_network_latents(payload):
    rows = []
    for atlas in sorted(payload):
        maps = payload[atlas]
        if not isinstance(maps, dict):
            continue
        for map_name in sorted(maps):
            latent = torch.as_tensor(maps[map_name], dtype=torch.float32).reshape(-1)
            if latent.numel() != latent_images.shape[1]:
                raise ValueError(f"{atlas}/{map_name} has shape {tuple(latent.shape)}, expected {latent_images.shape[1]}")
            rows.append((atlas, map_name, latent))
    return rows

imgs = load_dataset("networks")
masker = load_masker()
network_payload = load_latent("networks_neuro")
network_eval_set = flatten_network_latents(network_payload)
network_raw = torch.stack([row[2] for row in network_eval_set]).float()


# %%
@torch.no_grad()
def image_to_qformer_inputs(masked_image_tensor, *, canonical_banks, semantic_basis, projection_temp, raw_image_input_mode):
    raw_latent = enc(masked_image_tensor.to(device)).detach().float()
    raw_latent = raw_latent.reshape(1, -1) if raw_latent.ndim == 1 else raw_latent
    image_sem = F.normalize(proj_head_image(raw_latent).detach().float(), dim=1)
    image_input = project_to_canonical(get_canonical_bank(canonical_banks, semantic_basis), image_sem, temp=projection_temp).cpu()
    raw_input = raw_latent.cpu() if raw_image_input_mode == "raw" else image_input
    return raw_input, image_input

@torch.no_grad()
def generate_caption(
    raw_img,
    sem_img=None,
    *,
    max_new_tokens=256,
    num_beams=3,
    do_sample=False,
    temperature=None,
    top_p=None,
    seed=SEED,
    canonical_banks=None,
    semantic_basis=None,
    projection_temp=None,
    raw_image_input_mode=None,
    prefix_text=None,
):
    qformer.eval()
    model.eval()

    raw_img = torch.as_tensor(raw_img, dtype=torch.float32).reshape(1, -1)

    if sem_img is None:
        if canonical_banks is None or semantic_basis is None or projection_temp is None or raw_image_input_mode is None:
            raise ValueError(
                "Pass canonical_banks, semantic_basis, projection_temp, and "
                "raw_image_input_mode when sem_img is None."
            )
        image_sem = F.normalize(proj_head_image(raw_img.to(device)).detach().float(), dim=1)
        image_input = project_to_canonical(
            get_canonical_bank(canonical_banks, semantic_basis),
            image_sem,
            temp=projection_temp,
        ).cpu()
        raw_img = raw_img.cpu() if raw_image_input_mode == "raw" else image_input
        sem_img = image_input
    else:
        sem_img = torch.as_tensor(sem_img, dtype=torch.float32).reshape(1, -1)

    raw_img = raw_img.to(device=device, dtype=train_dtype)
    sem_img = sem_img.to(device=device, dtype=train_dtype)
    raw_img, sem_img = apply_input_policy(raw_img, sem_img)

    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
        vis = qformer(raw_img, sem_img)
        vis = match_lm_token_norm(vis).to(model_dtype)

        inputs_embeds = vis
        if prefix_text:
            prefix_ids = tokenizer(
                prefix_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].to(device)
            prefix_embeds = model.get_input_embeddings()(prefix_ids).to(model_dtype)
            inputs_embeds = torch.cat([vis, prefix_embeds], dim=1)

        attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

        gen_kwargs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            repetition_penalty=1.18,
            no_repeat_ngram_size=4,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            top_k=None,
        )
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

        dev = torch.device(device)
        cuda_devices = [torch.cuda.current_device() if dev.index is None else dev.index] if dev.type == "cuda" else []

        with torch.random.fork_rng(devices=cuda_devices, enabled=seed is not None):
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            out_ids = model.generate(**gen_kwargs)

    generated = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
    return (prefix_text or "") + generated


# %%
refs = [
    ["Laird", "Visual1"],
    ["Glasser", "Language"],
    ['Shirer', 'Auditory'],
    ["WashU", "LateralSM"],
    ["Du", "FPN-A"],
    ["YeoLab", "DefaultA"],
    ["WashU", "Salience"],
]

image_tensors = []
for item in refs:
    if len(item) == 1:
        continue
    atlas, map_name = item

    im = nib.Nifti1Image(imgs[atlas][map_name]["array"] / imgs[atlas][map_name]["array"].max(), imgs[atlas][map_name]["affine"])

    t = torch.from_numpy(masker.transform(resample_to_img(im, masker.mask_img, interpolation="nearest")))

    image_tensors.append(t)

# %%
do_sample = False
temperature = None
top_p=None
num_beams = 1
max_new_tokens = 256
generations = []
idx = 0

for ((atlas_name, map_name), t) in zip(refs, image_tensors):

    print(f"{atlas_name}_{map_name}")

    for seed, basis in enumerate(["network", "region", "function"]):

        with torch.no_grad():
            raw_latent, sem_latent = image_to_qformer_inputs(
                t.cuda(),
                canonical_banks=canonical_semantic_banks,
                semantic_basis=basis,
                projection_temp=0.05,
                raw_image_input_mode=RAW_IMAGE_INPUT_MODE,
            )
            pred = generate_caption(
                raw_latent,
                sem_latent,
                max_new_tokens=max_new_tokens,
                seed=seed,
                do_sample=do_sample,
                num_beams=num_beams,
                temperature=temperature,
                top_p=top_p,
                prefix_text=f"[{basis.upper()}]"
            )

            generations.append((atlas_name, map_name, pred))

            print(pred)
            print()

    idx += 1
    print()

# %%
torch.save(canonical_semantic_banks, CANONICAL_BANKS_PATH)

# %%
do_sample = False
temperature = None
top_p=None
num_beams = 5
max_new_tokens = 256
generations = []

idx = 0

for ((atlas_name, map_name), t) in zip(refs, image_tensors):

    print(f"{atlas_name}_{map_name}")

    for seed, basis in enumerate(["network", "region", "function"]):


        with torch.no_grad():
            raw_latent, sem_latent = image_to_qformer_inputs(
                t.cuda(),
                canonical_banks=canonical_semantic_banks,
                semantic_basis=basis,
                projection_temp=0.05,
                raw_image_input_mode=RAW_IMAGE_INPUT_MODE,
            )
            pred = generate_caption(
                raw_latent,
                sem_latent,
                max_new_tokens=max_new_tokens,
                seed=seed,
                do_sample=do_sample,
                num_beams=num_beams,
                temperature=temperature,
                top_p=top_p,
                prefix_text=f"[{basis.upper()}]"
            )

            generations.append((atlas_name, map_name, pred))

            print(pred)
            print()

    idx += 1
    print()

# %%
refs = [(i[0], i[1]) for i in network_eval_set]

image_tensors = []
for item in refs:
    if len(item) == 1:
        continue
    atlas, map_name = item

    im = nib.Nifti1Image(imgs[atlas][map_name]["array"] / imgs[atlas][map_name]["array"].max(), imgs[atlas][map_name]["affine"])

    t = torch.from_numpy(masker.transform(resample_to_img(im, masker.mask_img, interpolation="nearest")))

    image_tensors.append(t)

# %%
do_sample = False
temperature = None
top_p=None
num_beams = 5
max_new_tokens = 256

generations = []

idx = 0

for ((atlas_name, map_name), t) in zip(refs, image_tensors):

    print(f"{atlas_name}_{map_name}")

    for seed, basis in enumerate(["network", "region", "function"]):


        with torch.no_grad():
            raw_latent, sem_latent = image_to_qformer_inputs(
                t.cuda(),
                canonical_banks=canonical_semantic_banks,
                semantic_basis=basis,
                projection_temp=0.05,
                raw_image_input_mode=RAW_IMAGE_INPUT_MODE,
            )
            pred = generate_caption(
                raw_latent,
                sem_latent,
                max_new_tokens=max_new_tokens,
                seed=seed,
                do_sample=do_sample,
                num_beams=num_beams,
                temperature=temperature,
                top_p=top_p,
                prefix_text=f"[{basis.upper()}]"
            )

            generations.append((atlas_name, map_name, pred))

            print(pred)
            print()

    idx += 1
    print()
