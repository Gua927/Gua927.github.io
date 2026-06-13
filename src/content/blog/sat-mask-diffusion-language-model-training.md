---
title: "SAT-Mask: Self-Aligned Trajectory Masking for Diffusion Language Models"
date: 2026-06-13
excerpt: A short note on why random masking misaligns diffusion language model training with inference trajectories, and how SAT-Mask addresses it.
category: PUB-NOTE
tags: ["MDLM", "Order"]
author: Runze Tian
affiliation: GenSI Lab, THU-AIR
---

## Train-inference mismatch

Masked diffusion language models are trained to denoise partially masked text. The usual random-masking objective is simple, but it asks the model to learn from arbitrary mask patterns that do not match the structured denoising paths used at inference time.

Our paper, **Efficient Diffusion Language Model Training via Self-Aligned Trajectory Masking**, studies this train-inference mismatch and proposes **SAT-Mask**, a **S**elf-**A**ligned **T**rajectory **Masking** schedule.

## Self-aligned trajectory masking

SAT-Mask constructs training states with dynamic over-noising followed by margin-based partial denoising. The goal is to expose the model to contexts that follow an intrinsic easy-to-hard generation order, without changing the model architecture.

## Results

Across benchmarks, SAT-Mask improves both quality and efficiency: it improves Sudoku and Countdown accuracy, improves MAUVE on OpenWebText with lower GenPPL, and enables a smaller 170M model to match a 336M baseline on GSM8K with fewer training steps.
