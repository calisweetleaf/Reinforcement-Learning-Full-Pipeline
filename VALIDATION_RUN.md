(.venv) PS C:\Users\treyr\Desktop\Dev-Drive\Full-RLHF-Pipeline> python full_pipeline.py  
======================================================================
Full RLHF Pipeline Examples
======================================================================

======================================================================
Method Comparison
======================================================================

DPO:
  memory: Medium (2 models)
  speed: Fast
  stability: High
  best_for: General preference learning

GRPO:
  memory: Medium (2 models)
  speed: Medium
  stability: High
  best_for: Reasoning with verifiable rewards

PPO:
  memory: High (4 models)
  speed: Slow
  stability: Medium
  best_for: Maximum control, online learning

SimPO:
  memory: Low (1 model)
  speed: Fastest
  stability: Medium
  best_for: Resource-constrained training

KTO:
  memory: Medium (2 models)
  speed: Fast
  stability: High
  best_for: Unpaired preference data



======================================================================
Full RLHF Pipeline - Method: DPO
======================================================================
[01/31/26 01:52:15] INFO     Device: cpu, dtype: torch.bfloat16, AMP: False
[01/31/26 01:52:16] INFO     RLHFOrchestrator initialized with base model: gpt2
                    INFO     Output directory: rlhf_output
                    INFO     Self-improvement enabled: True

----------------------------------------------------------------------
STAGE 1: Supervised Fine-Tuning (SFT)
----------------------------------------------------------------------
Training on 3 examples...
Config: LR=5e-06, Epochs=2
[SFT complete]

----------------------------------------------------------------------
STAGE 3: Policy Optimization (DPO)
----------------------------------------------------------------------
Training with DPO...
Config: DPOConfig(learning_rate=5e-07, batch_size=2, num_epochs=2, weight_decay=0.01, max_grad_norm=1.0, warmup_ratio=0.1, warmup_steps=None, gradient_accumulation_steps=1, seed=42, fp16=False, bf16=True, use_amp=False, amp_dtype='bfloat16', logging_steps=10, eval_steps=100, save_steps=500, output_dir='./checkpoints/dpo', save_total_limit=3, resume_from_checkpoint=None, use_wandb=True, wandb_project='rlhf-training', wandb_run_name=None, use_tensorboard=False, tensorboard_dir=None, use_lora=False, lora_r=16, lora_alpha=32, lora_dropout=0.05, lora_target_modules=None, early_stopping_patience=None, early_stopping_threshold=0.0, beta=0.1, reference_model_freeze=True, label_smoothing=0.0, loss_type='sigmoid')
[DPO training complete]

----------------------------------------------------------------------
STAGE 4: Self-Play Enhancement
----------------------------------------------------------------------
Running self-play for capability improvement...
  - Generating synthetic training data
  - Iterative refinement
  - Capability testing
[Self-play complete]

----------------------------------------------------------------------
STAGE 5: Evaluation
----------------------------------------------------------------------
Evaluating on 2 test prompts...
Evaluation metrics:
  - KL Divergence: [would be computed]
  - Reward Accuracy: [would be computed]
  - Response Diversity: [would be computed]
  - Win Rate: [would be computed]

----------------------------------------------------------------------
Saving Final Model
----------------------------------------------------------------------
Model saved to ./final_model_dpo

======================================================================
Pipeline Complete!
======================================================================

Press Enter to continue to next method...



======================================================================
Full RLHF Pipeline - Method: GRPO
======================================================================
[01/31/26 01:52:30] INFO     Device: cpu, dtype: torch.bfloat16, AMP: False
[01/31/26 01:52:31] INFO     RLHFOrchestrator initialized with base model: gpt2
                    INFO     Output directory: rlhf_output
                    INFO     Self-improvement enabled: True

----------------------------------------------------------------------
STAGE 1: Supervised Fine-Tuning (SFT)
----------------------------------------------------------------------
Training on 3 examples...
Config: LR=5e-06, Epochs=2
[SFT complete]

----------------------------------------------------------------------
STAGE 3: Policy Optimization (GRPO)
----------------------------------------------------------------------
Training with GRPO...
Config: GRPOConfig(learning_rate=1e-06, batch_size=2, num_epochs=2, weight_decay=0.01, max_grad_norm=1.0, warmup_ratio=0.1, warmup_steps=None, gradient_accumulation_steps=1, seed=42, fp16=False, bf16=True, use_amp=False, amp_dtype='bfloat16', logging_steps=10, eval_steps=100, save_steps=500, output_dir='./checkpoints/grpo', save_total_limit=3, resume_from_checkpoint=None, use_wandb=True, wandb_project='rlhf-training', wandb_run_name=None, use_tensorboard=False, tensorboard_dir=None, use_lora=False, lora_r=16, lora_alpha=32, lora_dropout=0.05, lora_target_modules=None, early_stopping_patience=None, early_stopping_threshold=0.0, group_size=4, kl_coeff=0.1, clip_ratio=0.2, num_policy_updates=1, max_completion_length=512, temperature=1.0, use_verifiable_rewards=True)
[GRPO training complete]

----------------------------------------------------------------------
STAGE 4: Self-Play Enhancement
----------------------------------------------------------------------
Running self-play for capability improvement...
  - Generating synthetic training data
  - Iterative refinement
  - Capability testing
[Self-play complete]

----------------------------------------------------------------------
STAGE 5: Evaluation
----------------------------------------------------------------------
Evaluating on 2 test prompts...
Evaluation metrics:
  - KL Divergence: [would be computed]
  - Reward Accuracy: [would be computed]
  - Response Diversity: [would be computed]
  - Win Rate: [would be computed]

----------------------------------------------------------------------
Saving Final Model
----------------------------------------------------------------------
Model saved to ./final_model_grpo

======================================================================
Pipeline Complete!
======================================================================

Press Enter to continue to next method...



======================================================================
Full RLHF Pipeline - Method: SIMPO
======================================================================
[01/31/26 01:52:36] INFO     Device: cpu, dtype: torch.bfloat16, AMP: False
[01/31/26 01:52:37] INFO     RLHFOrchestrator initialized with base model: gpt2
                    INFO     Output directory: rlhf_output
                    INFO     Self-improvement enabled: True

----------------------------------------------------------------------
STAGE 1: Supervised Fine-Tuning (SFT)
----------------------------------------------------------------------
Training on 3 examples...
Config: LR=5e-06, Epochs=2
[SFT complete]

----------------------------------------------------------------------
STAGE 3: Policy Optimization (SIMPO)
----------------------------------------------------------------------
Training with SIMPO...
Config: SimPOConfig(learning_rate=5e-07, batch_size=2, num_epochs=2, weight_decay=0.01, max_grad_norm=1.0, warmup_ratio=0.1, warmup_steps=None, gradient_accumulation_steps=1, seed=42, fp16=False, bf16=True, use_amp=False, amp_dtype='bfloat16', logging_steps=10, eval_steps=100, save_steps=500, output_dir='./checkpoints/simpo', save_total_limit=3, resume_from_checkpoint=None, use_wandb=True, wandb_project='rlhf-training', wandb_run_name=None, use_tensorboard=False, tensorboard_dir=None, use_lora=False, lora_r=16, lora_alpha=32, lora_dropout=0.05, lora_target_modules=None, early_stopping_patience=None, early_stopping_threshold=0.0, beta=2.0, gamma=0.5, label_smoothing=0.0)
[SIMPO training complete]

----------------------------------------------------------------------
STAGE 4: Self-Play Enhancement
----------------------------------------------------------------------
Running self-play for capability improvement...
  - Generating synthetic training data
  - Iterative refinement
  - Capability testing
[Self-play complete]

----------------------------------------------------------------------
STAGE 5: Evaluation
----------------------------------------------------------------------
Evaluating on 2 test prompts...
Evaluation metrics:
  - KL Divergence: [would be computed]
  - Reward Accuracy: [would be computed]
  - Response Diversity: [would be computed]
  - Win Rate: [would be computed]

----------------------------------------------------------------------
Saving Final Model
----------------------------------------------------------------------
Model saved to ./final_model_simpo

======================================================================
Pipeline Complete!
======================================================================

Press Enter to continue to next method...



======================================================================
All Examples Complete!
======================================================================

Next steps:
1. Replace sample data with your own datasets
2. Adjust hyperparameters for your use case
3. Enable actual training by uncommenting orchestrator calls
4. Monitor training with WandB/TensorBoard
5. Evaluate thoroughly before deployment
(.venv) PS C:\Users\treyr\Desktop\Dev-Drive\Full-RLHF-Pipeline> 