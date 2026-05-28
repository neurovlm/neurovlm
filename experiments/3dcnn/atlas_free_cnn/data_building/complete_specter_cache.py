#!/usr/bin/env python
"""Complete the SPECTER text embedding cache with missing texts."""

import sys
from pathlib import Path
import pandas as pd
import torch

# Add src to path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from neurovlm.retrieval_resources import _load_specter


def main():
    cache_dir = ROOT / "experiments/3dcnn/atlas_free_cnn/cache"
    text_pairs_path = cache_dir / "hf_atlas_free_cnn/atlas_free_cnn_text_pairs.parquet"
    cache_path = cache_dir / "text_embeddings/specter_text_cache.pt"

    print("Loading text pairs...")
    text_pairs = pd.read_parquet(text_pairs_path)
    unique_texts = sorted(set(text_pairs['text'].unique()))
    print(f"Total unique texts in dataset: {len(unique_texts)}")

    print("\nLoading existing SPECTER cache...")
    existing_cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    print(f"Existing cache has {len(existing_cache)} embeddings")

    # Find missing texts
    cached_texts = set(existing_cache.keys())
    missing_texts = [t for t in unique_texts if t not in cached_texts]
    print(f"\nMissing {len(missing_texts)} texts from cache")

    if len(missing_texts) == 0:
        print("✓ Cache is complete!")
        return

    # Show breakdown by source
    missing_df = text_pairs[text_pairs['text'].isin(missing_texts)][['text', 'source']].drop_duplicates('text')
    print("\nMissing texts by source:")
    print(missing_df['source'].value_counts())

    # Load SPECTER model
    print("\nLoading SPECTER model...")
    specter = _load_specter()

    # Encode missing texts in batches
    print(f"\nEncoding {len(missing_texts)} missing texts...")
    batch_size = 32
    new_embeddings = {}

    for i in range(0, len(missing_texts), batch_size):
        batch = missing_texts[i:i+batch_size]
        print(f"  Batch {i//batch_size + 1}/{(len(missing_texts)-1)//batch_size + 1}: encoding {len(batch)} texts...")

        with torch.no_grad():
            embeddings = specter(batch)  # Returns [batch_size, 768]

        for text, emb in zip(batch, embeddings):
            new_embeddings[text] = emb.cpu()

    # Merge with existing cache
    print(f"\nMerging {len(new_embeddings)} new embeddings with existing cache...")
    complete_cache = {**existing_cache, **new_embeddings}
    print(f"Complete cache has {len(complete_cache)} embeddings")

    # Verify completeness
    still_missing = [t for t in unique_texts if t not in complete_cache]
    if len(still_missing) > 0:
        print(f"\n✗ ERROR: Still missing {len(still_missing)} texts!")
        return

    # Save updated cache
    backup_path = cache_path.with_suffix('.pt.backup')
    print(f"\nBacking up original cache to {backup_path.name}...")
    cache_path.rename(backup_path)

    print(f"Saving complete cache to {cache_path.name}...")
    torch.save(complete_cache, cache_path)

    print("\n✓✓✓ SUCCESS ✓✓✓")
    print(f"Complete SPECTER cache saved with {len(complete_cache)} embeddings")
    print(f"Original cache backed up to {backup_path}")
    print("\nNext steps:")
    print("1. Upload the updated specter_text_cache.pt to HuggingFace")
    print("2. Replace the file in the neurovlm/atlas_free_cnn_dataset repo")


if __name__ == "__main__":
    main()
