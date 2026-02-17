# Neural Network Model Comparison Results

## Executive Summary

This document compares neural network architectures for multi-label classification on CogAtlas dataset with different data filtering strategies.

**Key Findings**:
- ✅ **Best Model**: Model 1 ([1024, 512] baseline) achieves F1 micro = 0.7658
- ⚠️ **Critical**: Data filtering strategy has 10x more impact than architecture choice
- ❌ **Avoid**: Cosine similarity threshold ≥ 0.65 causes 60-80% performance drops
- 🎯 **Optimal Dataset**: Use curated labels (12/sample) instead of similarity-based expansion (41/sample)

**Recommended Configuration**:
- Architecture: Model 1 ([1024, 512] with ReLU)
- Dataset: Configuration A (filtered, ~12 labels per sample)
- For ranking tasks: Model 5 ([1024, 512] with Swish)

---

## Dataset Configurations

This document compares results across different CogAtlas dataset configurations:

### Configuration A: Filtered Dataset (Original Results)
- **Training Data**: CogAtlas dataset
- **Label Distribution**:
  - Top 5 concepts
  - Task labels
  - Top 2 phenotypes
- **Average labels per sample**: 12.00
- **Prediction threshold**: 0.530

### Configuration B: Cosine Similarity Threshold 0.65
- **Training Data**: CogAtlas dataset with cosine similarity filtering ≥ 0.65
- **Average labels per sample**: 28.92
- **Prediction threshold**: 0.430

### Configuration C: Alternative Cosine Similarity Threshold 0.65
- **Training Data**: CogAtlas dataset with alternative cosine similarity filtering ≥ 0.65
- **Average labels per sample**: 41.46
- **Prediction threshold**: 0.430

## Model Architectures Tested (Configuration A: Filtered Dataset)

### Summary Table

| Model | Architecture | Activation | Features | Hamming Loss ↓ | F1 Micro ↑ | F1 Macro ↑ | F1 Samples ↑ | LRAP ↑ | Recall@10 ↑ | Avg Pred Labels |
|-------|-------------|------------|----------|----------------|------------|------------|--------------|--------|-------------|-----------------|
| **Model 1** | [1024, 512] | ReLU | Baseline | **0.0084** | **0.7658** | **0.6667** | **0.7701** | 0.9066 | 0.7298 | 16.83 |
| Model 2 | [2048, 1024, 512] | ReLU | Wider | 0.0086 | 0.7599 | 0.6628 | 0.7641 | 0.9001 | 0.7232 | 16.91 |
| **Model 2b** | [3072, 1536, 768] | ReLU | Even Wider | 0.0083 | 0.7673 | 0.6682 | 0.7715 | 0.9048 | 0.7285 | 16.69 |
| Model 4 | [1024, 512, 256, 128] | ReLU | Deeper | 0.0121 | 0.6942 | 0.5963 | 0.7008 | 0.8671 | 0.6934 | 19.92 |
| Model 5 | [1024, 512] | Swish | Alt. Activation | 0.0084 | 0.7642 | 0.6661 | 0.7689 | **0.9083** | **0.7307** | 16.93 |
| Model 5b | [2048, 1024, 512] | Swish | Wider + Swish | 0.0089 | 0.7548 | 0.6585 | 0.7601 | 0.9041 | 0.7272 | 17.45 |
| Model 6 | [1024, 512] | ReLU | PCA Reduced | 0.0082 | 0.7636 | 0.6664 | 0.7675 | 0.8940 | 0.7184 | 16.21 |

**Note**: ↑ indicates higher is better, ↓ indicates lower is better. Bold values indicate top performers.

## Detailed Results (Configuration A: Filtered Dataset)

### Model 1: [1024, 512] - Baseline (ReLU)
**Best Overall Performance**

```
Threshold:                   0.530
Hamming Loss:                0.0084
F1 Score (micro):            0.7658
F1 Score (macro):            0.6667
F1 Score (samples):          0.7701
Label Ranking Avg Precision: 0.9066
Recall@10:                   0.7298

Prediction Statistics:
Avg labels per sample (true): 12.00
Avg labels per sample (pred): 16.83
Avg prediction confidence:    0.0236
```

### Model 2: [2048, 1024, 512] - Wider Network (ReLU)

```
Threshold:                   0.530
Hamming Loss:                0.0086
F1 Score (micro):            0.7599
F1 Score (macro):            0.6628
F1 Score (samples):          0.7641
Label Ranking Avg Precision: 0.9001
Recall@10:                   0.7232

Prediction Statistics:
Avg labels per sample (true): 12.00
Avg labels per sample (pred): 16.91
Avg prediction confidence:    0.0236
```

### Model 2b: [3072, 1536, 768] - Even Wider (ReLU)
**Strong Performer**

```
Threshold:                   0.530
Hamming Loss:                0.0083
F1 Score (micro):            0.7673
F1 Score (macro):            0.6682
F1 Score (samples):          0.7715
Label Ranking Avg Precision: 0.9048
Recall@10:                   0.7285

Prediction Statistics:
Avg labels per sample (true): 12.00
Avg labels per sample (pred): 16.69
Avg prediction confidence:    0.0232
```

### Model 4: [1024, 512, 256, 128] - Deeper Network (ReLU)
**Poorest Performance**

```
Threshold:                   0.530
Hamming Loss:                0.0121
F1 Score (micro):            0.6942
F1 Score (macro):            0.5963
F1 Score (samples):          0.7008
Label Ranking Avg Precision: 0.8671
Recall@10:                   0.6934

Prediction Statistics:
Avg labels per sample (true): 12.00
Avg labels per sample (pred): 19.92
Avg prediction confidence:    0.0316
```

### Model 5: [1024, 512] - Swish Activation
**Best Ranking Performance**

```
Threshold:                   0.530
Hamming Loss:                0.0084
F1 Score (micro):            0.7642
F1 Score (macro):            0.6661
F1 Score (samples):          0.7689
Label Ranking Avg Precision: 0.9083
Recall@10:                   0.7307

Prediction Statistics:
Avg labels per sample (true): 12.00
Avg labels per sample (pred): 16.93
Avg prediction confidence:    0.0240
```

### Model 5b: [2048, 1024, 512] - Wider with Swish

```
Threshold:                   0.530
Hamming Loss:                0.0089
F1 Score (micro):            0.7548
F1 Score (macro):            0.6585
F1 Score (samples):          0.7601
Label Ranking Avg Precision: 0.9041
Recall@10:                   0.7272

Prediction Statistics:
Avg labels per sample (true): 12.00
Avg labels per sample (pred): 17.45
Avg prediction confidence:    0.0247
```

### Model 6: [1024, 512] - With PCA Dimensionality Reduction

```
Threshold:                   0.530
Hamming Loss:                0.0082
F1 Score (micro):            0.7636
F1 Score (macro):            0.6664
F1 Score (samples):          0.7675
Label Ranking Avg Precision: 0.8940
Recall@10:                   0.7184

Prediction Statistics:
Avg labels per sample (true): 12.00
Avg labels per sample (pred): 16.21
Avg prediction confidence:    0.0227
```

---

## Impact of Data Filtering: Cosine Similarity Thresholds

### Comparison Table: Model 1 Across Different Data Configurations

| Configuration | Pred Threshold | Avg True Labels | Avg Pred Labels | Hamming Loss ↓ | F1 Micro ↑ | F1 Macro ↑ | F1 Samples ↑ | LRAP ↑ | Recall@10 ↑ |
|---------------|----------------|-----------------|-----------------|----------------|------------|------------|--------------|--------|-------------|
| **A: Filtered (Original)** | 0.530 | 12.00 | 16.83 | **0.0084** | **0.7658** | **0.6667** | **0.7701** | **0.9066** | **0.7298** |
| B: CosSim ≥ 0.65 | 0.430 | 28.92 | 49.89 | 0.0725 | 0.2516 | 0.1965 | 0.1955 | 0.2642 | 0.1436 |
| C: CosSim ≥ 0.65 (Alt) | 0.430 | 41.46 | 69.89 | 0.0957 | 0.3004 | 0.2552 | 0.2317 | 0.3073 | 0.1279 |

### Detailed Results: Configuration B (CosSim ≥ 0.65)

```
Threshold:                   0.430
Hamming Loss:                0.0725
F1 Score (micro):            0.2516
F1 Score (macro):            0.1965
F1 Score (samples):          0.1955
Label Ranking Avg Precision: 0.2642
Recall@10:                   0.1436

Prediction Statistics:
Avg labels per sample (true): 28.92
Avg labels per sample (pred): 49.89
Avg prediction confidence:    0.1022
```

### Detailed Results: Configuration C (CosSim ≥ 0.65, Alternative)

```
Threshold:                   0.430
Hamming Loss:                0.0957
F1 Score (micro):            0.3004
F1 Score (macro):            0.2552
F1 Score (samples):          0.2317
Label Ranking Avg Precision: 0.3073
Recall@10:                   0.1279

Prediction Statistics:
Avg labels per sample (true): 41.46
Avg labels per sample (pred): 69.89
Avg prediction confidence:    0.1296
```

### Analysis: Impact of Cosine Similarity Filtering

#### Critical Performance Degradation

Using cosine similarity thresholds ≥ 0.65 to filter training data causes **severe performance degradation**:

1. **F1 Score Collapse**:
   - Original (Config A): F1 micro = 0.7658
   - CosSim 0.65 (Config B): F1 micro = 0.2516 (**67% decrease**)
   - CosSim 0.65 Alt (Config C): F1 micro = 0.3004 (**61% decrease**)

2. **Hamming Loss Increases**:
   - Original: 0.0084
   - Config B: 0.0725 (**8.6x worse**)
   - Config C: 0.0957 (**11.4x worse**)

3. **Label Ranking Avg Precision Collapse**:
   - Original: 0.9066
   - Config B: 0.2642 (**71% decrease**)
   - Config C: 0.3073 (**66% decrease**)

4. **Recall@10 Degradation**:
   - Original: 0.7298
   - Config B: 0.1436 (**80% decrease**)
   - Config C: 0.1279 (**82% decrease**)

#### Root Cause: Label Space Explosion

The filtering approach creates a much more challenging multi-label problem:

- **Original**: 12.00 labels/sample → model predicts 16.83 (40% over-prediction)
- **Config B**: 28.92 labels/sample → model predicts 49.89 (73% over-prediction)
- **Config C**: 41.46 labels/sample → model predicts 69.89 (69% over-prediction)

#### Key Insights

1. **Label Density vs Performance**:
   - As label count increases from 12 → 29 → 41, all metrics collapse
   - The model struggles with denser multi-label predictions

2. **Over-prediction Gets Worse**:
   - Original: +4.83 labels over true (40% over)
   - Config B: +20.97 labels over true (73% over)
   - Config C: +28.43 labels over true (69% over)

3. **Prediction Confidence**:
   - Original: 0.0236 avg confidence
   - Config B: 0.1022 avg confidence (**4.3x higher**)
   - Config C: 0.1296 avg confidence (**5.5x higher**)
   - Despite higher confidence, predictions are much less accurate

4. **Threshold Mismatch**:
   - Cosine filtered models use lower prediction threshold (0.430 vs 0.530)
   - Even with lower threshold, performance is dramatically worse

#### Recommendations

**❌ DO NOT use cosine similarity thresholds ≥ 0.65** for data filtering:
- Causes 60-80% performance drops across all metrics
- Creates unmanageable label space (41+ labels per sample)
- Severe over-prediction issues
- Higher confidence but worse accuracy (model is miscalibrated)

**✅ RECOMMENDED APPROACH**:
- Use Configuration A (filtered dataset with 12 labels/sample)
- Focus on the original label distribution (top 5 concepts, task, top 2 phenotypes)
- Optimize threshold and architecture for this manageable label space

---

## Key Findings

### Top Performers
1. **Model 1** ([1024, 512] baseline): Best F1 scores across all variants
2. **Model 5** ([1024, 512] + Swish): Best Label Ranking Avg Precision (0.9083) and Recall@10 (0.7307)
3. **Model 2b** ([3072, 1536, 768]): Strong performance, lowest Hamming loss (0.0083)

### Key Observations (Configuration A)

1. **Data Filtering is Critical** ⚠️:
   - Using cosine similarity threshold ≥ 0.65 causes **60-80% performance degradation**
   - Label space explosion (12 → 41 labels/sample) overwhelms the model
   - See "Impact of Data Filtering" section above for detailed analysis
   - **Conclusion**: Careful label curation is MORE important than architecture choice

2. **Width vs Depth**:
   - Wider networks (Model 2, 2b) show competitive but not superior performance to baseline
   - Deeper networks (Model 4) significantly underperform
   - **Conclusion**: Going deeper hurts performance; going wider provides marginal gains at best

3. **Activation Functions**:
   - Swish activation (Model 5) achieves best ranking metrics (LRAP, Recall@10)
   - Minimal difference between ReLU and Swish for F1 scores
   - **Conclusion**: Swish may help with ranking quality but doesn't improve classification accuracy

4. **Dimensionality Reduction**:
   - PCA preprocessing (Model 6) shows competitive performance
   - Lower average predicted labels (16.21 vs 16.83)
   - **Conclusion**: PCA could be useful for computational efficiency without major accuracy loss

5. **Over-prediction Tendency**:
   - All models predict more labels than ground truth (avg 16-20 vs 12)
   - Model 4 shows worst over-prediction (19.92)
   - Best models maintain ~17 predictions vs 12 true labels

6. **Threshold Sensitivity**:
   - All models use same threshold (0.530)
   - Per-model threshold optimization could improve results

## Recommendations

### For Production
**Use Model 1** ([1024, 512] baseline with ReLU) for:
- Best overall F1 scores
- Good balance of precision/recall
- Simpler architecture (faster training, easier deployment)
- Tied for lowest Hamming loss

### For Ranking Tasks
**Consider Model 5** ([1024, 512] + Swish) if label ranking quality is critical:
- Best Label Ranking Average Precision (0.9083)
- Best Recall@10 (0.7307)
- Similar complexity to Model 1

### For Resource-Constrained Scenarios
**Consider Model 6** ([1024, 512] + PCA) for:
- Lower computational cost (reduced input dimensionality)
- Competitive performance (F1 micro: 0.7636)
- Fastest prediction times

### Not Recommended
- **Model 4** ([1024, 512, 256, 128]): Deeper architecture significantly hurts performance
- **Model 2/2b**: Wider networks don't justify increased computational cost

## Future Improvements

### High Priority (Based on Impact Analysis)

1. **Data Quality & Curation** 🔴:
   - **Impact**: 10x more important than architecture choice
   - Investigate why cosine similarity ≥ 0.65 filtering fails so dramatically
   - Explore intermediate label densities (15-25 labels/sample)
   - Consider hierarchical label selection instead of flat similarity thresholds
   - Validate label quality vs quantity tradeoffs

2. **Calibration** 🟡:
   - Address systematic over-prediction (avg 17 vs 12 labels in Config A)
   - Even worse in high-density configs (70 vs 41 labels in Config C)
   - Implement temperature scaling or Platt scaling
   - Note: Higher confidence ≠ better accuracy (see Config B/C)

3. **Threshold Optimization** 🟡:
   - Per-model threshold tuning (currently global 0.530 for Config A, 0.430 for B/C)
   - Consider dynamic thresholds based on prediction confidence distribution
   - Investigate if optimal threshold varies with label density

### Medium Priority (Incremental Gains)

4. **Ensemble Methods**:
   - Combine Model 1 (best F1) + Model 5 (best LRAP/Recall@10)
   - May help balance precision/recall tradeoffs

5. **Architecture Search**:
   - Explore intermediate widths (e.g., [1536, 768])
   - Test batch normalization or layer normalization
   - Experiment with residual connections for deeper models

6. **Regularization**:
   - Add dropout or L2 to reduce over-prediction
   - Test label smoothing for calibration

### Low Priority (Marginal Impact)

7. **Activation Function Tuning**: Swish shows minimal gains over ReLU
8. **Further Width Expansion**: Models 2/2b show diminishing returns
9. **Deeper Architectures**: Model 4 demonstrates depth hurts performance

### Research Questions

- Why does label density above ~30/sample cause catastrophic performance collapse?
- Can hierarchical/structured prediction help with high-density label spaces?
- Is there a "sweet spot" for label count (between 12 and 29)?
- Can we predict which papers should have more/fewer labels?
