"""Generate classification training data from brain images or text.

This script generates training data by searching CogAtlas similarity with higher k values
to get more training labels per paper. It can use either brain images or text embeddings.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Tuple, Optional
from scipy.sparse import csr_matrix


def generate_train_data_from_brain(
    output_dir: str = "docs/02_models",
    k_concepts: int = 5,
    k_disorders: int = 2,
    k_tasks: int = 5,
    save_matrices: bool = True
) -> Tuple[pd.DataFrame, Optional[csr_matrix], Optional[np.ndarray], Optional[np.ndarray]]:
    """Generate classification training data from brain images.

    This uses brain image embeddings to find similar CogAtlas terms and generates
    training matrices for term classification.

    Parameters
    ----------
    output_dir : str
        Directory to save output files
    k_concepts : int
        Number of top concepts per paper
    k_disorders : int
        Number of top disorders per paper
    k_tasks : int
        Number of top tasks per paper
    save_matrices : bool
        Whether to save term_matrix.npy, term_labels.npy, and term_pmids.npy

    Returns
    -------
    df : DataFrame
        DataFrame with pmid and cogatlas categories
    term_matrix : csr_matrix or None
        Sparse matrix of term occurrences (n_papers x n_unique_terms)
    term_labels : ndarray or None
        Array of unique term labels
    term_pmids : ndarray or None
        Array of PMIDs corresponding to rows in term_matrix
    """
    from neurovlm.retrieval_resources import _load_latent_neuro
    from neurovlm.core import NeuroVLM
    import torch

    print("="*70)
    print("GENERATING TRAINING DATA FROM BRAIN IMAGES")
    print("="*70)
    print(f"\nTarget k values:")
    print(f"  Concepts:  {k_concepts}")
    print(f"  Disorders: {k_disorders}")
    print(f"  Tasks:     {k_tasks}")
    print(f"  Total per paper: {k_concepts + k_disorders + k_tasks}")

    # Load brain vectors
    print("\nLoading brain vectors...")
    brain_vectors, pmids = _load_latent_neuro()
    print(f"Loaded {len(pmids)} brain vectors")

    # Initialize NeuroVLM with CogAtlas datasets
    print("\nInitializing NeuroVLM...")
    nvlm = NeuroVLM(datasets=["cogatlas", "cogatlas_disorder", "cogatlas_task"])

    # Process each paper
    print("\nSearching CogAtlas for all papers...")
    results = []
    all_terms_per_paper = []  # Collect all terms for matrix generation

    for i in tqdm(range(len(pmids)), desc="Processing papers"):
        brain_vector = brain_vectors[i:i+1]  # Keep as 2D
        pmid = pmids[i]

        try:
            # Search concepts
            concepts_result = nvlm.brain(brain_vector).to_text(datasets=["cogatlas"])
            concepts_df = concepts_result.top_k(k=k_concepts)
            top_concepts = concepts_df["title"].tolist()

            # Search disorders
            disorders_result = nvlm.brain(brain_vector).to_text(datasets=["cogatlas_disorder"])
            disorders_df = disorders_result.top_k(k=k_disorders)
            top_disorders = disorders_df["title"].tolist()

            # Search tasks
            tasks_result = nvlm.brain(brain_vector).to_text(datasets=["cogatlas_task"])
            tasks_df = tasks_result.top_k(k=k_tasks)
            top_tasks = tasks_df["title"].tolist()

            results.append({
                'pmid': int(pmid),
                'cogatlas_concepts': np.array(top_concepts),
                'cogatlas_disorder': np.array(top_disorders),
                'cogatlas_task': np.array(top_tasks)
            })

            # Collect all terms for this paper
            all_terms = list(top_concepts) + list(top_disorders) + list(top_tasks)
            all_terms_per_paper.append(all_terms)

        except Exception as e:
            print(f"\nWarning: Error processing PMID {pmid}: {e}")
            # Add empty entry
            results.append({
                'pmid': int(pmid),
                'cogatlas_concepts': np.array([]),
                'cogatlas_disorder': np.array([]),
                'cogatlas_task': np.array([])
            })
            all_terms_per_paper.append([])

    # Create DataFrame
    df = pd.DataFrame(results)

    print(f"\n\nGenerated dataset:")
    print(f"  Shape: {df.shape}")
    print(f"  Papers: {len(df)}")

    # Check actual k values
    sample_concepts = df['cogatlas_concepts'].iloc[0]
    sample_disorders = df['cogatlas_disorder'].iloc[0]
    sample_tasks = df['cogatlas_task'].iloc[0]

    print(f"\nActual terms per paper (sample):")
    print(f"  Concepts: {len(sample_concepts)}")
    print(f"  Disorders: {len(sample_disorders)}")
    print(f"  Tasks: {len(sample_tasks)}")

    # Generate term matrix if requested
    term_matrix = None
    term_labels = None
    term_pmids = None

    if save_matrices:
        print("\nGenerating term occurrence matrix...")

        # Get unique terms
        all_unique_terms = sorted(set(term for terms in all_terms_per_paper for term in terms))
        term_to_idx = {term: idx for idx, term in enumerate(all_unique_terms)}

        print(f"  Unique terms: {len(all_unique_terms)}")

        # Build sparse matrix
        rows = []
        cols = []
        data = []

        for paper_idx, terms in enumerate(all_terms_per_paper):
            for term in terms:
                rows.append(paper_idx)
                cols.append(term_to_idx[term])
                data.append(1)

        term_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(pmids), len(all_unique_terms)),
            dtype=np.int8
        )
        term_labels = np.array(all_unique_terms)
        term_pmids = pmids

        print(f"  Matrix shape: {term_matrix.shape}")
        print(f"  Matrix density: {term_matrix.nnz / (term_matrix.shape[0] * term_matrix.shape[1]):.4f}")

    # Save files
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    parquet_file = output_path / "paper_cogatlas_categories_brain.parquet"
    print(f"\nSaving DataFrame to {parquet_file}...")
    df.to_parquet(parquet_file, index=False)

    if save_matrices:
        matrix_file = output_path / "term_matrix_brain.npy"
        labels_file = output_path / "term_labels_brain.npy"
        pmids_file = output_path / "term_pmids_brain.npy"

        print(f"Saving term matrix to {matrix_file}...")
        np.save(matrix_file, term_matrix.toarray())

        print(f"Saving term labels to {labels_file}...")
        np.save(labels_file, term_labels)

        print(f"Saving term PMIDs to {pmids_file}...")
        np.save(pmids_file, term_pmids)

    print("Done!")

    return df, term_matrix, term_labels, term_pmids


def generate_train_data_from_text(
    output_dir: str = "docs/02_models",
    k_concepts: int = 5,
    k_disorders: int = 2,
    k_tasks: int = 5,
    save_matrices: bool = True
) -> Tuple[pd.DataFrame, Optional[csr_matrix], Optional[np.ndarray], Optional[np.ndarray]]:
    """Generate classification training data from text embeddings.

    This uses text (title + abstract) embeddings to find similar CogAtlas terms and generates
    training matrices for term classification.

    Parameters
    ----------
    output_dir : str
        Directory to save output files
    k_concepts : int
        Number of top concepts per paper
    k_disorders : int
        Number of top disorders per paper
    k_tasks : int
        Number of top tasks per paper
    save_matrices : bool
        Whether to save term_matrix.npy, term_labels.npy, and term_pmids.npy

    Returns
    -------
    df : DataFrame
        DataFrame with pmid and cogatlas categories
    term_matrix : csr_matrix or None
        Sparse matrix of term occurrences (n_papers x n_unique_terms)
    term_labels : ndarray or None
        Array of unique term labels
    term_pmids : ndarray or None
        Array of PMIDs corresponding to rows in term_matrix
    """
    from neurovlm.retrieval_resources import _load_latent_text
    from neurovlm.core import NeuroVLM
    import torch

    print("="*70)
    print("GENERATING TRAINING DATA FROM TEXT EMBEDDINGS")
    print("="*70)
    print(f"\nTarget k values:")
    print(f"  Concepts:  {k_concepts}")
    print(f"  Disorders: {k_disorders}")
    print(f"  Tasks:     {k_tasks}")
    print(f"  Total per paper: {k_concepts + k_disorders + k_tasks}")

    # Load text embeddings
    print("\nLoading text embeddings...")
    text_embeddings, pmids = _load_latent_text()
    print(f"Loaded {len(pmids)} text embeddings")

    # Initialize NeuroVLM with CogAtlas datasets
    print("\nInitializing NeuroVLM...")
    nvlm = NeuroVLM(datasets=["cogatlas", "cogatlas_disorder", "cogatlas_task"])

    # Process each paper
    print("\nSearching CogAtlas for all papers...")
    results = []
    all_terms_per_paper = []  # Collect all terms for matrix generation

    for i in tqdm(range(len(pmids)), desc="Processing papers"):
        text_embedding = text_embeddings[i]  # Already 1D tensor
        pmid = pmids[i]

        try:
            # Search concepts using text embedding
            concepts_result = nvlm.text(text_embedding).to_text(datasets=["cogatlas"])
            concepts_df = concepts_result.top_k(k=k_concepts)
            top_concepts = concepts_df["title"].tolist()

            # Search disorders
            disorders_result = nvlm.text(text_embedding).to_text(datasets=["cogatlas_disorder"])
            disorders_df = disorders_result.top_k(k=k_disorders)
            top_disorders = disorders_df["title"].tolist()

            # Search tasks
            tasks_result = nvlm.text(text_embedding).to_text(datasets=["cogatlas_task"])
            tasks_df = tasks_result.top_k(k=k_tasks)
            top_tasks = tasks_df["title"].tolist()

            results.append({
                'pmid': int(pmid),
                'cogatlas_concepts': np.array(top_concepts),
                'cogatlas_disorder': np.array(top_disorders),
                'cogatlas_task': np.array(top_tasks)
            })

            # Collect all terms for this paper
            all_terms = list(top_concepts) + list(top_disorders) + list(top_tasks)
            all_terms_per_paper.append(all_terms)

        except Exception as e:
            print(f"\nWarning: Error processing PMID {pmid}: {e}")
            # Add empty entry
            results.append({
                'pmid': int(pmid),
                'cogatlas_concepts': np.array([]),
                'cogatlas_disorder': np.array([]),
                'cogatlas_task': np.array([])
            })
            all_terms_per_paper.append([])

    # Create DataFrame
    df = pd.DataFrame(results)

    print(f"\n\nGenerated dataset:")
    print(f"  Shape: {df.shape}")
    print(f"  Papers: {len(df)}")

    # Check actual k values
    sample_concepts = df['cogatlas_concepts'].iloc[0]
    sample_disorders = df['cogatlas_disorder'].iloc[0]
    sample_tasks = df['cogatlas_task'].iloc[0]

    print(f"\nActual terms per paper (sample):")
    print(f"  Concepts: {len(sample_concepts)}")
    print(f"  Disorders: {len(sample_disorders)}")
    print(f"  Tasks: {len(sample_tasks)}")

    # Generate term matrix if requested
    term_matrix = None
    term_labels = None
    term_pmids = None

    if save_matrices:
        print("\nGenerating term occurrence matrix...")

        # Get unique terms
        all_unique_terms = sorted(set(term for terms in all_terms_per_paper for term in terms))
        term_to_idx = {term: idx for idx, term in enumerate(all_unique_terms)}

        print(f"  Unique terms: {len(all_unique_terms)}")

        # Build sparse matrix
        rows = []
        cols = []
        data = []

        for paper_idx, terms in enumerate(all_terms_per_paper):
            for term in terms:
                rows.append(paper_idx)
                cols.append(term_to_idx[term])
                data.append(1)

        term_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(pmids), len(all_unique_terms)),
            dtype=np.int8
        )
        term_labels = np.array(all_unique_terms)
        term_pmids = pmids

        print(f"  Matrix shape: {term_matrix.shape}")
        print(f"  Matrix density: {term_matrix.nnz / (term_matrix.shape[0] * term_matrix.shape[1]):.4f}")

    # Save files
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    parquet_file = output_path / "paper_cogatlas_categories_text.parquet"
    print(f"\nSaving DataFrame to {parquet_file}...")
    df.to_parquet(parquet_file, index=False)

    if save_matrices:
        matrix_file = output_path / "term_matrix_text.npy"
        labels_file = output_path / "term_labels_text.npy"
        pmids_file = output_path / "term_pmids_text.npy"

        print(f"Saving term matrix to {matrix_file}...")
        np.save(matrix_file, term_matrix.toarray())

        print(f"Saving term labels to {labels_file}...")
        np.save(labels_file, term_labels)

        print(f"Saving term PMIDs to {pmids_file}...")
        np.save(pmids_file, term_pmids)

    print("Done!")

    return df, term_matrix, term_labels, term_pmids


def main():
    parser = argparse.ArgumentParser(
        description="Generate classification training data from brain images or text"
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['brain', 'text', 'both'],
        default='brain',
        help='Generate from brain images, text embeddings, or both'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='docs/02_models',
        help='Output directory for generated files'
    )
    parser.add_argument(
        '--k-concepts',
        type=int,
        default=5,
        help='Number of top concepts per paper'
    )
    parser.add_argument(
        '--k-disorders',
        type=int,
        default=2,
        help='Number of top disorders per paper'
    )
    parser.add_argument(
        '--k-tasks',
        type=int,
        default=5,
        help='Number of top tasks per paper'
    )
    parser.add_argument(
        '--no-matrices',
        action='store_true',
        help='Skip generating term_matrix, term_labels, and term_pmids files'
    )

    args = parser.parse_args()

    save_matrices = not args.no_matrices

    if args.mode in ['brain', 'both']:
        print("\n" + "="*70)
        print("GENERATING FROM BRAIN IMAGES")
        print("="*70)
        generate_train_data_from_brain(
            output_dir=args.output_dir,
            k_concepts=args.k_concepts,
            k_disorders=args.k_disorders,
            k_tasks=args.k_tasks,
            save_matrices=save_matrices
        )

    if args.mode in ['text', 'both']:
        print("\n" + "="*70)
        print("GENERATING FROM TEXT EMBEDDINGS")
        print("="*70)
        generate_train_data_from_text(
            output_dir=args.output_dir,
            k_concepts=args.k_concepts,
            k_disorders=args.k_disorders,
            k_tasks=args.k_tasks,
            save_matrices=save_matrices
        )

    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"\nGenerated files saved to: {args.output_dir}")
    print("\nNext steps:")
    print("  1. Use the generated files for training")
    print("  2. Train classification models with the term matrices")


if __name__ == "__main__":
    main()
