"""Fine-grained hierarchical brain region mapping.

This module provides an even more granular hierarchy, creating ~100-150 categories
by further splitting high-frequency intermediate terms by laterality and subregions.
"""

import numpy as np
import json
from typing import Dict, List, Set, Tuple
from pathlib import Path


# Fine-grained brain region taxonomy
# Even more granular than intermediate, focusing on splitting high-frequency terms
BRAIN_REGION_HIERARCHY_FINE = {
    # FRONTAL LOBE - Very detailed splits
    "left dorsolateral prefrontal cortex": [
        "left dorsolateral prefrontal", "left dlpfc", "left ba9", "left ba46",
        "left dorsolateral", "l dlpfc", "l dorsolateral prefrontal"
    ],

    "right dorsolateral prefrontal cortex": [
        "right dorsolateral prefrontal", "right dlpfc", "right ba9", "right ba46",
        "right dorsolateral", "r dlpfc", "r dorsolateral prefrontal"
    ],

    "dorsolateral prefrontal cortex ba9": [
        "dlpfc ba9", "ba9", "middle frontal gyrus ba9"
    ],

    "dorsolateral prefrontal cortex ba46": [
        "dlpfc ba46", "ba46", "middle frontal gyrus ba46"
    ],

    # Split VLPFC by laterality and subregion
    "left ventrolateral prefrontal cortex ba44": [
        "left ba44", "left pars opercularis", "l ba44", "left vlpfc ba44"
    ],

    "left ventrolateral prefrontal cortex ba45": [
        "left ba45", "left pars triangularis", "l ba45", "left vlpfc ba45"
    ],

    "left ventrolateral prefrontal cortex ba47": [
        "left ba47", "left pars orbitalis", "l ba47", "left vlpfc ba47"
    ],

    "right ventrolateral prefrontal cortex ba44": [
        "right ba44", "right pars opercularis", "r ba44", "right vlpfc ba44"
    ],

    "right ventrolateral prefrontal cortex ba45": [
        "right ba45", "right pars triangularis", "r ba45", "right vlpfc ba45"
    ],

    "right ventrolateral prefrontal cortex ba47": [
        "right ba47", "right pars orbitalis", "r ba47", "right vlpfc ba47"
    ],

    "left inferior frontal gyrus": [
        "left inferior frontal gyrus", "left ifg", "l inferior frontal", "l ifg"
    ],

    "right inferior frontal gyrus": [
        "right inferior frontal gyrus", "right ifg", "r inferior frontal", "r ifg"
    ],

    "left ventromedial prefrontal cortex": [
        "left ventromedial prefrontal", "left vmpfc", "left ba10", "left ba11",
        "l vmpfc", "l ventromedial prefrontal"
    ],

    "right ventromedial prefrontal cortex": [
        "right ventromedial prefrontal", "right vmpfc", "right ba10", "right ba11",
        "r vmpfc", "r ventromedial prefrontal"
    ],

    "left orbitofrontal cortex": [
        "left orbitofrontal", "left ofc", "l ofc", "l orbitofrontal"
    ],

    "right orbitofrontal cortex": [
        "right orbitofrontal", "right ofc", "r ofc", "r orbitofrontal"
    ],

    "medial orbitofrontal cortex": [
        "medial ofc", "medial orbitofrontal", "ba11"
    ],

    "lateral orbitofrontal cortex": [
        "lateral ofc", "lateral orbitofrontal", "ba47"
    ],

    # ACC - Very detailed
    "rostral anterior cingulate cortex": [
        "rostral anterior cingulate", "rostral acc", "racc", "ba32"
    ],

    "dorsal anterior cingulate cortex": [
        "dorsal anterior cingulate", "dorsal acc", "dacc", "ba24"
    ],

    "subgenual anterior cingulate cortex": [
        "subgenual anterior cingulate", "subgenual acc", "sgacc", "subgenual", "ba25"
    ],

    "left rostral anterior cingulate cortex": [
        "left rostral anterior cingulate", "left rostral acc", "left racc"
    ],

    "right rostral anterior cingulate cortex": [
        "right rostral anterior cingulate", "right rostral acc", "right racc"
    ],

    "left dorsal anterior cingulate cortex": [
        "left dorsal anterior cingulate", "left dorsal acc", "left dacc"
    ],

    "right dorsal anterior cingulate cortex": [
        "right dorsal anterior cingulate", "right dorsal acc", "right dacc"
    ],

    "left anterior cingulate cortex": [
        "left anterior cingulate", "left acc", "l acc"
    ],

    "right anterior cingulate cortex": [
        "right anterior cingulate", "right acc", "r acc"
    ],

    "left medial frontal cortex": [
        "left medial frontal", "left mfc", "left ba8", "l medial frontal",
        "left dorsomedial prefrontal", "left dmpfc"
    ],

    "right medial frontal cortex": [
        "right medial frontal", "right mfc", "right ba8", "r medial frontal",
        "right dorsomedial prefrontal", "right dmpfc"
    ],

    "left middle frontal gyrus": [
        "left middle frontal", "left middle frontal gyrus", "left mfg"
    ],

    "right middle frontal gyrus": [
        "right middle frontal", "right middle frontal gyrus", "right mfg"
    ],

    # SFG - Already split, keep
    "left superior frontal gyrus": [
        "left superior frontal", "left superior frontal gyrus", "left sfg",
        "l superior frontal", "l sfg"
    ],

    "right superior frontal gyrus": [
        "right superior frontal", "right superior frontal gyrus", "right sfg",
        "r superior frontal", "r sfg"
    ],

    "left frontal pole": [
        "left frontal pole", "left ba10", "left frontopolar", "l frontal pole"
    ],

    "right frontal pole": [
        "right frontal pole", "right ba10", "right frontopolar", "r frontal pole"
    ],

    "left premotor cortex": [
        "left premotor", "left pmc", "left ba6", "l premotor"
    ],

    "right premotor cortex": [
        "right premotor", "right pmc", "right ba6", "r premotor"
    ],

    "left supplementary motor area": [
        "left supplementary motor", "left sma", "left pre-sma", "l sma"
    ],

    "right supplementary motor area": [
        "right supplementary motor", "right sma", "right pre-sma", "r sma"
    ],

    # Primary motor - Split by laterality (already exists, keep)
    "left primary motor cortex": [
        "left primary motor", "left motor cortex", "left m1", "left ba4",
        "left precentral gyrus", "left precentral", "l motor cortex", "l m1"
    ],

    "right primary motor cortex": [
        "right primary motor", "right motor cortex", "right m1", "right ba4",
        "right precentral gyrus", "right precentral", "r motor cortex", "r m1"
    ],

    # TEMPORAL LOBE - Detailed laterality splits
    "left superior temporal gyrus": [
        "left superior temporal", "left stg", "left ba22", "left ba41", "left ba42",
        "l superior temporal", "l stg"
    ],

    "right superior temporal gyrus": [
        "right superior temporal", "right stg", "right ba22", "right ba41", "right ba42",
        "r superior temporal", "r stg"
    ],

    "left middle temporal gyrus": [
        "left middle temporal", "left mtg", "left ba21", "left ba37",
        "l middle temporal", "l mtg"
    ],

    "right middle temporal gyrus": [
        "right middle temporal", "right mtg", "right ba21", "right ba37",
        "r middle temporal", "r mtg"
    ],

    "left inferior temporal gyrus": [
        "left inferior temporal", "left itg", "left ba20",
        "l inferior temporal", "l itg"
    ],

    "right inferior temporal gyrus": [
        "right inferior temporal", "right itg", "right ba20",
        "r inferior temporal", "r itg"
    ],

    "left temporal pole": [
        "left temporal pole", "left ba38", "l temporal pole"
    ],

    "right temporal pole": [
        "right temporal pole", "right ba38", "r temporal pole"
    ],

    "left fusiform gyrus": [
        "left fusiform", "left fusiform gyrus", "left ba37",
        "l fusiform", "left fusiform face area"
    ],

    "right fusiform gyrus": [
        "right fusiform", "right fusiform gyrus", "right ba37",
        "r fusiform", "right fusiform face area"
    ],

    # TEMPORAL - LIMBIC - Laterality splits
    "left hippocampus": [
        "left hippocampus", "left hippocampal", "left hc", "l hippocampus",
        "left ca1", "left ca2", "left ca3", "left dentate gyrus"
    ],

    "right hippocampus": [
        "right hippocampus", "right hippocampal", "right hc", "r hippocampus",
        "right ca1", "right ca2", "right ca3", "right dentate gyrus"
    ],

    "left parahippocampal gyrus": [
        "left parahippocampal", "left parahippocampal gyrus", "l parahippocampal"
    ],

    "right parahippocampal gyrus": [
        "right parahippocampal", "right parahippocampal gyrus", "r parahippocampal"
    ],

    "left entorhinal cortex": [
        "left entorhinal", "left entorhinal cortex", "left ba28", "l entorhinal"
    ],

    "right entorhinal cortex": [
        "right entorhinal", "right entorhinal cortex", "right ba28", "r entorhinal"
    ],

    "left amygdala": [
        "left amygdala", "left amygdaloid", "l amygdala",
        "left basolateral amygdala", "left central amygdala"
    ],

    "right amygdala": [
        "right amygdala", "right amygdaloid", "r amygdala",
        "right basolateral amygdala", "right central amygdala"
    ],

    # PARIETAL LOBE - Detailed splits
    "left superior parietal lobule": [
        "left superior parietal", "left spl", "left ba5", "left ba7",
        "l superior parietal"
    ],

    "right superior parietal lobule": [
        "right superior parietal", "right spl", "right ba5", "right ba7",
        "r superior parietal"
    ],

    # IPL - Split by both subregion AND laterality
    "left angular gyrus": [
        "left angular", "left angular gyrus", "left ba39", "l angular"
    ],

    "right angular gyrus": [
        "right angular", "right angular gyrus", "right ba39", "r angular"
    ],

    "left supramarginal gyrus": [
        "left supramarginal", "left supramarginal gyrus", "left ba40", "l supramarginal"
    ],

    "right supramarginal gyrus": [
        "right supramarginal", "right supramarginal gyrus", "right ba40", "r supramarginal"
    ],

    "left inferior parietal lobule": [
        "left inferior parietal", "left ipl", "l inferior parietal"
    ],

    "right inferior parietal lobule": [
        "right inferior parietal", "right ipl", "r inferior parietal"
    ],

    "left precuneus": [
        "left precuneus", "left ba7", "l precuneus"
    ],

    "right precuneus": [
        "right precuneus", "right ba7", "r precuneus"
    ],

    "left postcentral gyrus": [
        "left postcentral", "left postcentral gyrus", "left ba1", "left ba2", "left ba3",
        "left primary somatosensory", "left s1", "l postcentral", "l somatosensory"
    ],

    "right postcentral gyrus": [
        "right postcentral", "right postcentral gyrus", "right ba1", "right ba2", "right ba3",
        "right primary somatosensory", "right s1", "r postcentral", "r somatosensory"
    ],

    "left intraparietal sulcus": [
        "left intraparietal", "left ips", "l intraparietal", "l ips"
    ],

    "right intraparietal sulcus": [
        "right intraparietal", "right ips", "r intraparietal", "r ips"
    ],

    # OCCIPITAL LOBE - V1 already split, add more
    "left primary visual cortex": [
        "left primary visual", "left v1", "left ba17", "left striate",
        "left calcarine", "l v1", "l primary visual"
    ],

    "right primary visual cortex": [
        "right primary visual", "right v1", "right ba17", "right striate",
        "right calcarine", "r v1", "r primary visual"
    ],

    "left secondary visual cortex": [
        "left secondary visual", "left v2", "left ba18", "left extrastriate", "l v2"
    ],

    "right secondary visual cortex": [
        "right secondary visual", "right v2", "right ba18", "right extrastriate", "r v2"
    ],

    "left visual area v3": [
        "left v3", "left ba19", "left visual area v3", "l v3"
    ],

    "right visual area v3": [
        "right v3", "right ba19", "right visual area v3", "r v3"
    ],

    "left visual area v4": [
        "left v4", "left visual area v4", "l v4"
    ],

    "right visual area v4": [
        "right v4", "right visual area v4", "r v4"
    ],

    "left visual area v5": [
        "left v5", "left mt", "left middle temporal area", "l v5", "l mt"
    ],

    "right visual area v5": [
        "right v5", "right mt", "right middle temporal area", "r v5", "r mt"
    ],

    "left cuneus": [
        "left cuneus", "l cuneus"
    ],

    "right cuneus": [
        "right cuneus", "r cuneus"
    ],

    "left lingual gyrus": [
        "left lingual", "left lingual gyrus", "l lingual"
    ],

    "right lingual gyrus": [
        "right lingual", "right lingual gyrus", "r lingual"
    ],

    "left lateral occipital cortex": [
        "left lateral occipital", "left ba19", "l lateral occipital"
    ],

    "right lateral occipital cortex": [
        "right lateral occipital", "right ba19", "r lateral occipital"
    ],

    # CINGULATE - PCC already has left/right
    "left posterior cingulate cortex": [
        "left posterior cingulate", "left pcc", "l posterior cingulate", "l pcc"
    ],

    "right posterior cingulate cortex": [
        "right posterior cingulate", "right pcc", "r posterior cingulate", "r pcc"
    ],

    "left midcingulate cortex": [
        "left midcingulate", "left mcc", "l midcingulate"
    ],

    "right midcingulate cortex": [
        "right midcingulate", "right mcc", "r midcingulate"
    ],

    # INSULA - Already well split, keep existing
    "left anterior insula": [
        "left anterior insula", "left anterior insular", "l anterior insula", "lai"
    ],

    "right anterior insula": [
        "right anterior insula", "right anterior insular", "r anterior insula", "rai"
    ],

    "left posterior insula": [
        "left posterior insula", "left posterior insular", "l posterior insula", "lpi"
    ],

    "right posterior insula": [
        "right posterior insula", "right posterior insular", "r posterior insula", "rpi"
    ],

    # BASAL GANGLIA - Laterality splits
    "left caudate nucleus": [
        "left caudate", "left caudate nucleus", "l caudate",
        "left head of caudate", "left body of caudate", "left tail of caudate"
    ],

    "right caudate nucleus": [
        "right caudate", "right caudate nucleus", "r caudate",
        "right head of caudate", "right body of caudate", "right tail of caudate"
    ],

    "left putamen": [
        "left putamen", "l putamen", "left dorsal striatum"
    ],

    "right putamen": [
        "right putamen", "r putamen", "right dorsal striatum"
    ],

    "left nucleus accumbens": [
        "left nucleus accumbens", "left nacc", "left ventral striatum",
        "l nucleus accumbens", "l nacc"
    ],

    "right nucleus accumbens": [
        "right nucleus accumbens", "right nacc", "right ventral striatum",
        "r nucleus accumbens", "r nacc"
    ],

    "left globus pallidus": [
        "left globus pallidus", "left pallidum", "left gp", "l globus pallidus"
    ],

    "right globus pallidus": [
        "right globus pallidus", "right pallidum", "right gp", "r globus pallidus"
    ],

    "left substantia nigra": [
        "left substantia nigra", "left sn", "l substantia nigra"
    ],

    "right substantia nigra": [
        "right substantia nigra", "right sn", "r substantia nigra"
    ],

    # THALAMUS - Detailed
    "left thalamus": [
        "left thalamus", "left thalamic", "l thalamus"
    ],

    "right thalamus": [
        "right thalamus", "right thalamic", "r thalamus"
    ],

    "left mediodorsal thalamus": [
        "left mediodorsal thalamus", "left md thalamus", "l mediodorsal thalamus"
    ],

    "right mediodorsal thalamus": [
        "right mediodorsal thalamus", "right md thalamus", "r mediodorsal thalamus"
    ],

    "left pulvinar": [
        "left pulvinar", "left pulvinar nucleus", "l pulvinar"
    ],

    "right pulvinar": [
        "right pulvinar", "right pulvinar nucleus", "r pulvinar"
    ],

    "left lateral geniculate nucleus": [
        "left lateral geniculate", "left lgn", "l lgn"
    ],

    "right lateral geniculate nucleus": [
        "right lateral geniculate", "right lgn", "r lgn"
    ],

    "left medial geniculate nucleus": [
        "left medial geniculate", "left mgn", "l mgn"
    ],

    "right medial geniculate nucleus": [
        "right medial geniculate", "right mgn", "r mgn"
    ],

    # CEREBELLUM - Laterality
    "left cerebellum": [
        "left cerebellum", "left cerebellar", "l cerebellum"
    ],

    "right cerebellum": [
        "right cerebellum", "right cerebellar", "r cerebellum"
    ],

    "cerebellar vermis": [
        "vermis", "cerebellar vermis"
    ],

    "left cerebellar hemisphere": [
        "left cerebellar hemisphere", "left lateral cerebellum", "l cerebellar hemisphere"
    ],

    "right cerebellar hemisphere": [
        "right cerebellar hemisphere", "right lateral cerebellum", "r cerebellar hemisphere"
    ],

    # BRAINSTEM - Some laterality
    "midbrain": [
        "midbrain", "mesencephalon"
    ],

    "pons": [
        "pons", "pontine"
    ],

    "medulla": [
        "medulla", "medulla oblongata"
    ],

    "periaqueductal gray": [
        "periaqueductal", "periaqueductal gray", "pag"
    ],

    "left ventral tegmental area": [
        "left ventral tegmental", "left vta", "l vta"
    ],

    "right ventral tegmental area": [
        "right ventral tegmental", "right vta", "r vta"
    ],

    "locus coeruleus": [
        "locus coeruleus", "lc"
    ],
}


# Import the necessary functions from the hierarchy module
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from brain_region_hierarchy import (
    build_child_to_parent_mapping,
    collapse_to_hierarchical_regions,
)


def merge_cogatlas_with_fine_regions(
    cogatlas_matrix: np.ndarray,
    cogatlas_labels: List[str],
    brain_region_matrix: np.ndarray,
    brain_region_labels: List[str],
    min_papers_per_parent: int = 5
) -> Tuple[np.ndarray, List[str], Dict]:
    """Merge CogAtlas with fine-grained hierarchical brain regions.

    This uses the finest-grained hierarchy with maximum laterality and subregion splits.

    Parameters
    ----------
    cogatlas_matrix : np.ndarray
        Binary matrix of CogAtlas terms (n_papers, n_cogatlas_terms)
    cogatlas_labels : list
        CogAtlas term names
    brain_region_matrix : np.ndarray
        Binary matrix of brain region terms (n_papers, n_brain_regions)
    brain_region_labels : list
        Brain region term names
    min_papers_per_parent : int
        Minimum papers required for a parent category to be included

    Returns
    -------
    merged_matrix : np.ndarray
        Combined matrix with CogAtlas + fine hierarchical regions
    merged_labels : list
        Combined label names
    merge_info : dict
        Information about the merging process
    """
    # Use fine hierarchy
    collapsed_regions, parent_regions, mapping_info = \
        collapse_to_hierarchical_regions(
            brain_region_matrix,
            brain_region_labels,
            hierarchy=BRAIN_REGION_HIERARCHY_FINE,
            min_papers_per_parent=min_papers_per_parent
        )

    # Combine CogAtlas (unchanged) with collapsed regions
    merged_matrix = np.concatenate([cogatlas_matrix, collapsed_regions], axis=1)
    merged_labels = cogatlas_labels + parent_regions

    # Store merge info
    merge_info = {
        "n_cogatlas_terms": len(cogatlas_labels),
        "n_original_brain_regions": len(brain_region_labels),
        "n_collapsed_brain_regions": len(parent_regions),
        "n_total_terms": len(merged_labels),
        "merge_mapping": mapping_info['parent_to_children'],
        "parent_terms": parent_regions,
        "child_to_parent": mapping_info['child_to_parent'],
        "unmatched_terms": mapping_info['unmatched_terms'],
        "min_papers_per_parent": min_papers_per_parent,
        "hierarchy_type": "fine"
    }

    return merged_matrix, merged_labels, merge_info


def save_fine_hierarchical_dataset(
    matrix: np.ndarray,
    labels: List[str],
    pmids: np.ndarray,
    merge_info: Dict,
    output_prefix: str = "fine_hierarchical_term"
) -> None:
    """Save fine hierarchical dataset to files.

    Parameters
    ----------
    matrix : np.ndarray
        Term label matrix
    labels : list
        Term names
    pmids : np.ndarray
        PMID identifiers
    merge_info : dict
        Information about the merging process
    output_prefix : str
        Prefix for output files
    """
    import json

    # Save arrays
    np.save(f"{output_prefix}_matrix.npy", matrix)
    np.save(f"{output_prefix}_labels.npy", np.array(labels))
    np.save(f"{output_prefix}_pmids.npy", pmids)

    # Save merge info with proper structure
    with open(f"{output_prefix}_merge_info.json", 'w') as f:
        info_serializable = {
            'n_cogatlas_terms': merge_info['n_cogatlas_terms'],
            'n_original_brain_regions': merge_info['n_original_brain_regions'],
            'n_collapsed_brain_regions': merge_info['n_collapsed_brain_regions'],
            'n_total_terms': merge_info['n_total_terms'],
            'min_papers_per_parent': merge_info['min_papers_per_parent'],
            'hierarchy_type': merge_info['hierarchy_type'],
            'parent_terms': merge_info['parent_terms'],
            'merge_mapping': merge_info['merge_mapping'],
            'child_to_parent': merge_info['child_to_parent'],
            'unmatched_terms': merge_info['unmatched_terms']
        }
        json.dump(info_serializable, f, indent=2)

    print(f"\nSaved fine hierarchical dataset:")
    print(f"  {output_prefix}_matrix.npy")
    print(f"  {output_prefix}_labels.npy")
    print(f"  {output_prefix}_pmids.npy")
    print(f"  {output_prefix}_merge_info.json")
