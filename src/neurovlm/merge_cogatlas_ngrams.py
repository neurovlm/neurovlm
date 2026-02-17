"""Merge CogAtlas and N-gram term datasets.

This module merges the sparse CogAtlas term data with n-gram data:
1. Consolidates Networks and Regions into a single "brain_region" category
2. Selects top 2-5 most relevant brain region terms per paper from n-grams
3. Merges with CogAtlas concepts, disorders, and tasks
4. Creates a unified term matrix for classification

This addresses the sparsity issue in CogAtlas by adding complementary n-gram data.
"""

import numpy as np
import pandas as pd
import json
from typing import Tuple, List, Dict, Optional
from pathlib import Path
from collections import Counter


def merge_cogatlas_ngrams(
    cogatlas_threshold: Optional[float] = None,
    brain_region_terms_per_paper: int = 3,
    output_prefix: str = "docs/02_models/merged_term",
    save_files: bool = True
) -> Tuple[np.ndarray, List[str], np.ndarray, Dict[str, List[str]]]:
    """Merge CogAtlas and n-gram term datasets from HuggingFace.

    Parameters
    ----------
    cogatlas_threshold : float, optional
        If provided, loads threshold-filtered CogAtlas data (e.g., 0.6, 0.65)
        If None, loads standard CogAtlas data
    brain_region_terms_per_paper : int
        Number of brain region terms to select per paper from n-grams (2-5 recommended)
    output_prefix : str
        Prefix for output files
    save_files : bool
        Whether to save output files

    Returns
    -------
    merged_matrix : np.ndarray
        Binary term matrix (n_papers, n_merged_terms)
    merged_labels : list of str
        Combined term names
    aligned_pmids : np.ndarray
        Aligned PMIDs
    category_info : dict
        Category information with mappings
    """
    print("="*70)
    print("MERGING COGATLAS AND N-GRAM DATASETS")
    print("="*70)

    # Load CogAtlas data from HuggingFace
    print("\n1. Loading CogAtlas data from HuggingFace...")
    from neurovlm.retrieval_resources import _load_cogatlas_term_data, _load_cogatlas_term_threshold_data

    if cogatlas_threshold is not None:
        cogatlas_matrix, cogatlas_labels_arr, cogatlas_pmids, cogatlas_category_info = _load_cogatlas_term_threshold_data(cogatlas_threshold)
        print(f"   Using threshold-filtered data (threshold={cogatlas_threshold})")
    else:
        cogatlas_matrix, cogatlas_labels_arr, cogatlas_pmids, cogatlas_category_info = _load_cogatlas_term_data()

    cogatlas_labels = cogatlas_labels_arr.tolist() if hasattr(cogatlas_labels_arr, 'tolist') else list(cogatlas_labels_arr)

    print(f"   CogAtlas matrix: {cogatlas_matrix.shape}")
    print(f"   CogAtlas terms: {len(cogatlas_labels)}")
    print(f"   CogAtlas PMIDs: {len(cogatlas_pmids)}")

    # Load n-gram data from HuggingFace
    print("\n2. Loading n-gram data from HuggingFace...")
    from neurovlm.retrieval_resources import _load_ngram_data, _download_from_hf

    ngram_matrix, ngram_labels_arr = _load_ngram_data()
    ngram_labels = ngram_labels_arr.tolist() if hasattr(ngram_labels_arr, 'tolist') else list(ngram_labels_arr)

    # Load n-gram PMIDs and categories from HuggingFace
    pmids_path = _download_from_hf("neurovlm/cognitive_atlas", "pmids.txt")
    with open(pmids_path, 'r') as f:
        ngram_pmids = np.array([line.strip() for line in f if line.strip() != 'pmid'])

    categories_path = _download_from_hf("neurovlm/cognitive_atlas", "label_categories.json")
    with open(categories_path, 'r') as f:
        ngram_categories = json.load(f)

    print(f"   N-gram matrix: {ngram_matrix.shape}")
    print(f"   N-gram terms: {len(ngram_labels)}")
    print(f"   N-gram PMIDs: {len(ngram_pmids)}")

    # Find brain region terms (Networks + Regions)
    print("\n3. Identifying brain region terms from n-grams...")
    brain_region_indices = []
    brain_region_labels = []

    for i, label in enumerate(ngram_labels):
        if label in ngram_categories:
            cats = ngram_categories[label]
            # Consolidate Networks and Regions into brain_region
            if cats.get('Networks', False) or cats.get('Regions', False):
                brain_region_indices.append(i)
                brain_region_labels.append(label)

    print(f"   Found {len(brain_region_labels)} brain region terms")
    print(f"   Sample: {brain_region_labels[:10]}")

    # Align PMIDs
    print("\n4. Aligning PMIDs...")
    cogatlas_pmids_str = cogatlas_pmids.astype(str)
    ngram_pmids_str = ngram_pmids.astype(str)

    common_pmids = np.intersect1d(cogatlas_pmids_str, ngram_pmids_str)
    print(f"   CogAtlas PMIDs: {len(cogatlas_pmids)}")
    print(f"   N-gram PMIDs: {len(ngram_pmids)}")
    print(f"   Common PMIDs: {len(common_pmids)}")

    # Align matrices
    cogatlas_indices = np.array([np.where(cogatlas_pmids_str == pmid)[0][0] for pmid in common_pmids])
    ngram_indices = np.array([np.where(ngram_pmids_str == pmid)[0][0] for pmid in common_pmids])

    aligned_cogatlas_matrix = cogatlas_matrix[cogatlas_indices]
    aligned_ngram_matrix = ngram_matrix[ngram_indices]

    # Select brain region terms per paper from n-grams
    if brain_region_terms_per_paper == -1:
        print(f"\n5. Extracting ALL brain region terms per paper...")
        # Extract ALL brain regions (no limit)
        brain_region_matrix = aligned_ngram_matrix[:, brain_region_indices]
        selected_brain_matrix = brain_region_matrix.astype(bool)
        terms_added_per_paper = selected_brain_matrix.sum(axis=1)
    else:
        print(f"\n5. Selecting top {brain_region_terms_per_paper} brain region terms per paper...")
        # Extract brain region columns from n-gram matrix
        brain_region_matrix = aligned_ngram_matrix[:, brain_region_indices]

        # For each paper, select up to k terms (if available)
        # We'll create a new binary matrix with the selected terms
        num_papers = len(common_pmids)
        selected_brain_matrix = np.zeros((num_papers, len(brain_region_labels)), dtype=bool)

        terms_added_per_paper = []
        for paper_idx in range(num_papers):
            # Get indices of present brain region terms for this paper
            present_term_indices = np.where(brain_region_matrix[paper_idx])[0]

            # Select up to k terms
            # For now, we'll just take the first k (you could add ranking logic here)
            selected_indices = present_term_indices[:brain_region_terms_per_paper]

            # Mark selected terms as present
            selected_brain_matrix[paper_idx, selected_indices] = True
            terms_added_per_paper.append(len(selected_indices))

    print(f"   Brain region terms added per paper:")
    print(f"     Mean: {np.mean(terms_added_per_paper):.2f}")
    print(f"     Median: {np.median(terms_added_per_paper):.1f}")
    print(f"     Max: {np.max(terms_added_per_paper)}")
    print(f"     Papers with 0 terms: {np.sum(np.array(terms_added_per_paper) == 0)}")

    # Merge CogAtlas with selected brain region terms
    print("\n6. Merging datasets...")

    # Combined labels: CogAtlas + brain_region n-grams
    merged_labels = cogatlas_labels + brain_region_labels

    # Combined matrix
    merged_matrix = np.concatenate([aligned_cogatlas_matrix, selected_brain_matrix], axis=1)

    print(f"   Merged matrix shape: {merged_matrix.shape}")
    print(f"   Total terms: {len(merged_labels)}")
    print(f"     - CogAtlas: {len(cogatlas_labels)}")
    print(f"     - Brain regions (n-gram): {len(brain_region_labels)}")

    # Statistics
    terms_per_paper = merged_matrix.sum(axis=1)
    papers_per_term = merged_matrix.sum(axis=0)

    print(f"\n7. Merged dataset statistics:")
    print(f"   Terms per paper: mean={terms_per_paper.mean():.1f}, std={terms_per_paper.std():.1f}, "
          f"min={terms_per_paper.min()}, max={terms_per_paper.max()}")
    print(f"   Papers per term: mean={papers_per_term.mean():.1f}, std={papers_per_term.std():.1f}, "
          f"min={papers_per_term.min()}, max={papers_per_term.max()}")
    print(f"   Sparsity: {(merged_matrix == 0).sum() / merged_matrix.size:.2%}")

    # Original CogAtlas-only stats for comparison
    cogatlas_terms_per_paper = aligned_cogatlas_matrix.sum(axis=1)
    print(f"\n   Comparison with CogAtlas-only:")
    print(f"   CogAtlas terms/paper: mean={cogatlas_terms_per_paper.mean():.1f}")
    print(f"   Merged terms/paper:   mean={terms_per_paper.mean():.1f}")
    print(f"   Improvement: +{terms_per_paper.mean() - cogatlas_terms_per_paper.mean():.1f} terms/paper")

    # Category information (already loaded from HuggingFace)
    category_info = {
        'cogatlas_concepts': cogatlas_category_info.get('concepts', []),
        'cogatlas_disorders': cogatlas_category_info.get('disorders', []),
        'cogatlas_tasks': cogatlas_category_info.get('tasks', []),
        'brain_region': brain_region_labels,
        'term_to_category': {}
    }

    # Build term_to_category mapping
    for term in category_info['cogatlas_concepts']:
        category_info['term_to_category'][term] = 'cogatlas_concept'
    for term in category_info['cogatlas_disorders']:
        category_info['term_to_category'][term] = 'cogatlas_disorder'
    for term in category_info['cogatlas_tasks']:
        category_info['term_to_category'][term] = 'cogatlas_task'

    # Add brain region category mappings
    for term in brain_region_labels:
        category_info['term_to_category'][term] = 'brain_region'

    # Save files
    if save_files:
        print(f"\n8. Saving files...")
        np.save(f"{output_prefix}_matrix.npy", merged_matrix)
        np.save(f"{output_prefix}_labels.npy", np.array(merged_labels))
        np.save(f"{output_prefix}_pmids.npy", common_pmids)

        with open(f"{output_prefix}_category_info.json", 'w') as f:
            json.dump(category_info, f, indent=2)

        print(f"   {output_prefix}_matrix.npy")
        print(f"   {output_prefix}_labels.npy")
        print(f"   {output_prefix}_pmids.npy")
        print(f"   {output_prefix}_category_info.json")

    print("\n" + "="*70)
    print("MERGE COMPLETE!")
    print("="*70)

    return merged_matrix, merged_labels, common_pmids, category_info


def analyze_merged_distribution(
    merged_matrix: np.ndarray,
    merged_labels: List[str],
    category_info: Dict[str, List[str]]
) -> None:
    """Analyze the distribution of merged terms.

    Parameters
    ----------
    merged_matrix : np.ndarray
        Merged binary term matrix
    merged_labels : list of str
        Merged term names
    category_info : dict
        Category information
    """
    print("\n" + "="*70)
    print("MERGED TERM DISTRIBUTION ANALYSIS")
    print("="*70)

    papers_per_term = merged_matrix.sum(axis=0)

    # Most common terms overall
    sorted_indices = np.argsort(papers_per_term)

    print("\n=== Most Common Terms (all categories) ===")
    for i in sorted_indices[-20:][::-1]:
        term = merged_labels[i]
        count = papers_per_term[i]
        category = category_info['term_to_category'].get(term, 'unknown')
        print(f"  {term:<40} {count:>5} papers ({category})")

    # Category-specific statistics
    print("\n=== Category Statistics ===")

    categories = [
        ('CogAtlas Concepts', category_info.get('cogatlas_concepts', [])),
        ('CogAtlas Disorders', category_info.get('cogatlas_disorders', [])),
        ('CogAtlas Tasks', category_info.get('cogatlas_tasks', [])),
        ('Brain Regions (N-gram)', category_info.get('brain_region', []))
    ]

    for cat_name, cat_terms in categories:
        if not cat_terms:
            continue

        cat_indices = [merged_labels.index(t) for t in cat_terms if t in merged_labels]
        cat_counts = papers_per_term[cat_indices]

        print(f"\n{cat_name}:")
        print(f"  Terms: {len(cat_terms)}")
        print(f"  Papers per term: mean={cat_counts.mean():.1f}, std={cat_counts.std():.1f}")
        print(f"  Min: {cat_counts.min()}, Max: {cat_counts.max()}")

        # Show top terms in this category
        top_in_category = np.argsort(cat_counts)[-5:][::-1]
        print(f"  Top 5 terms:")
        for idx in top_in_category:
            term = cat_terms[idx]
            count = cat_counts[idx]
            print(f"    - {term}: {count} papers")


def main():
    """Main function to merge CogAtlas and n-gram datasets."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge CogAtlas and n-gram term datasets from HuggingFace"
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='CogAtlas similarity threshold for filtering (e.g., 0.6, 0.65). If not provided, uses standard CogAtlas data'
    )
    parser.add_argument(
        '--brain-region-terms',
        type=int,
        default=3,
        help='Number of brain region terms to add per paper from n-grams (2-5 recommended, -1 for ALL, default: 3)'
    )
    parser.add_argument(
        '--output-prefix',
        type=str,
        default='docs/02_models/merged_term',
        help='Prefix for output files'
    )
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Run detailed analysis of merged data'
    )

    args = parser.parse_args()

    # Merge datasets
    merged_matrix, merged_labels, pmids, category_info = merge_cogatlas_ngrams(
        cogatlas_threshold=args.threshold,
        brain_region_terms_per_paper=args.brain_region_terms,
        output_prefix=args.output_prefix,
        save_files=True
    )

    # Analyze if requested
    if args.analyze:
        analyze_merged_distribution(merged_matrix, merged_labels, category_info)

    print("\nNext steps:")
    print("  1. Train term-level classifier with merged data:")
    print("     python -m neurovlm.train_term_classification \\")
    print(f"       --term-labels {args.output_prefix}_matrix.npy \\")
    print(f"       --term-names {args.output_prefix}_labels.npy \\")
    print(f"       --term-pmids {args.output_prefix}_pmids.npy")


if __name__ == "__main__":
    main()
