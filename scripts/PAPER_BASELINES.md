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

We reproduce **all 8 scenes** (room0/1/2 + office0/1/2/3/4), running each
**once** (not 5×). Data for all 8 comes from vMAP's `vmap.zip` (the public
source the authors' data_generation pipeline renders from), paired with the
authors' pretrained segmentation head + `semantic_classes.pkl`; label
compatibility is validated per scene by `colab_setup_sni.sh` before any run.

Our 8-scene average is therefore directly comparable to the paper's 8-scene
average. The only intended departures from the paper: 1× per scene instead of
5× (so we report the mean over 8 single runs, not 40), and vMAP's distributed
renders standing in for the authors' unreleased office/room2 renders.

A successful "works as the authors intended" claim lands within ~30% of the
paper's numbers on the 8-scene aggregate. Larger gaps prompt a debug pass
(data layout, weights, env stack, hyperparameters).

## What to fill in after our runs land

| metric            | paper avg | our 8-scene avg | gap |
|-------------------|-----------|-----------------|-----|
| Depth L1 [cm]     | 0.766     | ?               | ?   |
| Acc. [cm]         | 1.942     | ?               | ?   |
| Comp. [cm]        | 1.702     | ?               | ?   |
| Comp. Ratio [%]   | 96.624    | ?               | ?   |
| ATE Mean [cm]     | 0.397     | ?               | ?   |
| ATE RMSE [cm]     | 0.456     | ?               | ?   |

To populate the "our 8-scene avg" column, average the per-scene values from
the `MyDrive/Outputs/sni_slam_replica/_eval/<scene>_metrics.txt` outputs
across all 8 scenes.

## Optional: per-scene targets (need supplementary)

Other published methods (NICE-SLAM, ESLAM, Co-SLAM) report per-scene Replica
numbers in their own supplementaries. We can cross-reference those if we
need a per-scene target without obtaining SNI-SLAM's supplementary.

The arXiv HTML (https://arxiv.org/html/2311.11016v3) at time of writing does
NOT include the supplementary material. Options if we want per-scene paper
numbers later:
1. Download supplementary PDF from the project page
2. Pull values from NICE-SLAM (parent method) supplementary as a proxy
