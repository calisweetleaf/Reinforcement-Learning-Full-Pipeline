## Model Merging and Ensembling

`model_merging.py` implements weight-level merging and ensembling utilities for combining multiple fine-tuned models into a single policy or for ensemble generation at inference time.

`MergeConfig` selects a merge method (`task_arithmetic`, `ties`, `slerp`, `dare`) and controls weight normalization and sparsity density. `ModelMerger` computes parameter deltas between a base model and fine-tuned variants, merges deltas according to the selected method, and applies the merged delta to the base state dict. TIES merging performs trim-elect-sign-merge over parameter deltas, SLERP interpolates between two models on a spherical path, and DARE drops and rescales a sparse subset of deltas.

`ModelSoup` supports uniform or weighted averaging across models, and a greedy soup mode that adds models only when they improve an evaluation function. This provides a practical path to merge multiple specializations without explicit task routing.

`EnsemblePolicy` supports average-logit decoding or simple voting across multiple models. The interface mirrors standard generation so it can be wrapped around existing policies with minimal changes.

`layer_wise_interpolation` enables per-layer weighting between two models, which is useful when preserving lower-layer behavior while tuning higher layers for task specialization.
