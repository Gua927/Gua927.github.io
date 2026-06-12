---
title: Efficient Diffusion Language Model Training via Self-Aligned Trajectory Masking
date: 2026-05-08
authors:
  - name: Runze Tian
    highlight: true
  - name: Zhilong Zhang
  - name: Yuxuan Song
  - name: Keyue Qiu
  - name: Hao Zhou
venue: In the 40th Conference on Neural Information Processing Systems, 2026
status: Under Review
image: /assets/img/SAT-Mask.png
imageAlt: SAT-Mask trajectory masking overview
blog: /blog/sat-mask-diffusion-language-model-training/
abstract: "Masked diffusion models (MDMs) enable flexible text generation, but standard random-masking training is misaligned with the structured denoising trajectories used at inference time. We show this mismatch induces exposure bias and dilutes model capacity by requiring the model to represent arbitrary mask patterns, incurring a structural information tax per token under a linear masking schedule. As this issue originates during training, it cannot be resolved by inference-time samplers alone. We propose SAT-Mask, a Self-Aligned Trajectory Masking schedule that aligns training with inference through a shared transition kernel. SAT-Mask constructs training states via dynamic over-noising followed by margin-based partial denoising, exposing contexts that follow an intrinsic easy-to-hard generation order without architectural changes. Across benchmarks, SAT-Mask improves both quality and efficiency: it achieves +62.4% / +16.0% accuracy on Sudoku and Countdown, improves MAUVE on OpenWebText by up to +112.5% with lower GenPPL, and enables a 170M model to match a 336M baseline on GSM8K (54.28% vs. 54.96%) using 28.3% fewer training steps, with up to 68.2% overall reduction in training steps."
---
