set -x

# ==============================================================================
# GRPO 训练 GSM8K + Qwen2-7B-Instruct
#
# 入口链路：
#   verl.trainer.main_ppo.main()  —— Hydra 装配 config/ppo_trainer.yaml
#     → run_ppo()                 —— 启动 Ray、spawn TaskRunner
#       → TaskRunner.run()        —— 注册 role → 建 tokenizer/dataset → RayPPOTrainer.fit()
#
# 本脚本所有行都是 Hydra CLI override，按 config 树前缀分组如下：
#   - algorithm.*           : 算法选择（adv_estimator 等）
#   - data.*                : 数据层（parquet 路径、batch、长度限制）
#   - actor_rollout_ref.*   : 三合一 worker 的子配置
#       .model.*   共享模型设置（path, gradient_checkpointing, remove_padding）
#       .actor.*   训练引擎（optim, PPO mini/micro batch, KL loss, fsdp_config）
#       .rollout.* 推理引擎（vllm/sglang 后端, TP size, n 组大小, gpu_mem）
#       .ref.*     参考策略（log_prob batch, fsdp offload）
#   - trainer.*             : 运行时（n_gpus, epochs, save/test freq, logger）
# ==============================================================================

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40 \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    algorithm.use_kl_in_reward=False \
    \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='qwen2_7b_function_rm' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@

# ------------------------------------------------------------------------------
# 关键字段速查（后续 M3/M4/M5 会频繁用到）：
#   rollout.n=5                      GRPO 组大小；rollout 后 batch 会从 1024 膨胀到 5120
#   rollout.tensor_model_parallel_size=2   rollout engine 的 TP（可独立于 actor TP）
#   rollout.gpu_memory_utilization=0.6     vLLM KV cache 上限 60%，留 40% 给 colocate actor 权重
#   actor.use_kl_loss=True + kl_loss_type=low_var_kl   GRPO paper 的 k3 KL 估计
#   actor.use_kl_loss + algorithm.use_kl_in_reward     二选一，不要同时加
#   ref.fsdp_config.param_offload=True     ref 权重默认 CPU，需要时再上卡，省显存
# ------------------------------------------------------------------------------
