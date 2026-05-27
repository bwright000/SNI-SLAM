# SNI-SLAM paper-reported Replica baselines

Source: SNI-SLAM CVPR 2024 paper (arXiv 2311.11016 v3), Table 1, Section 4.1
Experimental Results.

The paper reports **aggregate** metrics across all 8 Replica scenes (room0/1/2 +
office0/1/2/3/4), each averaged over 5 independent runs (40 runs total). The
per-scene breakdown is in the supplementary, which is not in the public arXiv HTML.

## Paper Table 1 — SNI-SLAM on Replica (aggregate)

| metric            | value     | note |
|-------------------|-----------|------|
| Depth L1 [cm]     | 0.766     | mean L1 depth error |
| Acc. [cm]         | 1.942     | reconstruction accuracy |
| Comp. [cm]        | 1.702     | reconstruction completeness |
| Comp. Ratio [%]   | 96.624    | % of GT points reconstructed within threshold |
| ATE Mean [cm]     | 0.397     | tracking mean translation error |
| ATE RMSE [cm]     | 0.456     | tracking root-mean-squared error |

> *"The results are average of 8 scenes on the Replica dataset. To ensure more
> objectivity in the results, each scene is tested and averaged with five
> independent runs."*

## What we are reproducing

User picked a subset of 4 scenes (room0, room1, office0, office1) — half of
the paper's 8 — and we will run each **once** (not 5×). So our comparison is
necessarily less rigorous, but the **average across our 4 scenes should land
in the ballpark** of the paper's 8-scene average if reproduction is successful.

A defensible reproduction would land within ~30% of the paper's numbers on
each metric on the aggregate of our subset. Larger gaps would prompt a
debug pass (data layout, weights, env stack, hyperparameters).

## What to fill in after our runs land

| metric            | paper avg | our subset avg | gap |
|-------------------|-----------|----------------|-----|
| Depth L1 [cm]     | 0.766     | ?              | ?   |
| Acc. [cm]         | 1.942     | ?              | ?   |
| Comp. [cm]        | 1.702     | ?              | ?   |
| Comp. Ratio [%]   | 96.624    | ?              | ?   |
| ATE Mean [cm]     | 0.397     | ?              | ?   |
| ATE RMSE [cm]     | 0.456     | ?              | ?   |

To populate the "our subset avg" column, average the per-scene values from
`scripts/_eval/<scene>_metrics.txt` outputs across the 4 scenes.

## Optional: per-scene targets (need supplementary)

Other published methods (NICE-SLAM, ESLAM, Co-SLAM) report per-scene Replica
numbers in their own supplementaries. We can cross-reference those if we
need a per-scene target without obtaining SNI-SLAM's supplementary.

The arXiv HTML (https://arxiv.org/html/2311.11016v3) at time of writing does
NOT include the supplementary material. Options if we want per-scene paper
numbers later:
1. Download supplementary PDF from the project page
2. Pull values from NICE-SLAM (parent method) supplementary as a proxy
