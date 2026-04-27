# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from enum import Enum

from omegaconf import DictConfig

from verl.single_controller.base import Worker
from verl.trainer.distillation import is_distillation_enabled
from verl.trainer.ppo.core_algos import AdvantageEstimator

WorkerType = type[Worker]


# What: Enum tagging each worker role (Actor, Rollout, ActorRollout[Ref], Critic,
#   RefPolicy, RewardModel, TeacherModel, Env). __str__ collapses to the string tag
#   used as WorkerGroup / checkpoint-subdir key (e.g. Role.Critic -> "critic").
# Lifecycle: module-import constants; instances are stored as keys in
#   role_worker_mapping and resource-pool mapping before init_workers() runs.
# Called by: TaskRunner.run() in main_ppo.py (registers Role->WorkerClass and
#   Role->pool), RayPPOTrainer.init_workers() / fit() in ray_trainer.py, and
#   str(Role.*) used as checkpoint path segments.
# Branches (fused vs split):
#   - ActorRollout: HybridEngine fuses actor training + vLLM rollout in one Ray
#     actor so weights reshard in-place (no 2x model copy).
#   - ActorRolloutRef: same fusion, also hosts the frozen ref policy (LoRA path,
#     ref_in_actor=True) to skip a separate RefPolicy worker.
#   - Critic / RefPolicy / RewardModel / TeacherModel: separate workers, each
#     optional — registered only when need_critic / need_reference_policy /
#     need_reward_model / need_teacher_policy return True.
# Why: Key abstraction for the single-controller: each Role maps to a resource pool
#   and a WorkerGroup that dispatch decorators look up by tag. ActorRollout/
#   ActorRolloutRef are fused (HybridEngine) so train and rollout reshard in-place,
#   avoiding a 2x copy.
class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6
    Env = 7
    TeacherModel = 8

    def __str__(self):
        return self._get_role_string()

    def _get_role_string(self):
        role_mapping = {
            Role.Actor: "actor",
            Role.Rollout: "rollout",
            Role.ActorRollout: "actor_rollout",
            Role.Critic: "critic",
            Role.RefPolicy: "ref",
            Role.RewardModel: "rm",
            Role.ActorRolloutRef: "actor_rollout_ref",
            Role.TeacherModel: "teacher",
        }
        return role_mapping.get(self, self.name.lower())

    @classmethod
    def from_string(cls, name: str):
        string_mapping = {
            "actor": cls.Actor,
            "rollout": cls.Rollout,
            "actor_rollout": cls.ActorRollout,
            "critic": cls.Critic,
            "ref": cls.RefPolicy,
            "rm": cls.RewardModel,
            "actor_rollout_ref": cls.ActorRolloutRef,
        }
        role = string_mapping.get(name.lower())
        if role is None:
            raise ValueError(f"No Role found for string: {name}")
        return role


# What: Predicate returning True iff π_ref must be instantiated for KL anchoring.
#   True when algorithm.use_kl_in_reward (KL added to per-token reward) OR
#   actor.use_kl_loss (KL folded into the actor loss term).
# Lifecycle: called once during TaskRunner.run() before worker registration, and
#   again inside RayPPOTrainer.__init__ to set self.use_reference_policy.
# Called by: TaskRunner.run() / add_ref_policy_worker in main_ppo.py,
#   RayPPOTrainer.__init__ in ray_trainer.py.
# Branches:
#   - use_kl_in_reward or use_kl_loss -> True, register Role.RefPolicy (or reuse
#     Role.ActorRolloutRef when ref_in_actor=True for LoRA).
#   - neither set (e.g. pure GRPO without KL, REINFORCE++ variants) -> False,
#     skip the ref worker entirely.
# Why: Ref policy is only materialized when KL anchoring is actually used — either
#   as a reward penalty or a direct actor-loss term. Avoids paying the extra model
#   copy when an algorithm (e.g. pure GRPO without KL loss) doesn't need a frozen
#   anchor.
def need_reference_policy(
    config: DictConfig,
) -> bool:
    """Given the config, do we need ref policy."""
    return config.algorithm.get("use_kl_in_reward", False) or config.actor_rollout_ref.actor.use_kl_loss


def need_teacher_policy(
    config: DictConfig,
) -> bool:
    """Given the config, do we need distillation policy."""
    return is_distillation_enabled(config.get("distillation"))


def need_reward_model(
    config: DictConfig,
) -> bool:
    """Given the config, do we need reward model."""
    return config.reward.reward_model.enable


# What: Predicate gating Critic worker instantiation. Explicit critic.enable
#   overrides; otherwise True iff adv_estimator == GAE. Emits a warning when
#   defaulting off so an unintended GRPO->PPO config typo is visible.
# Lifecycle: called once during TaskRunner.run() before worker registration, and
#   again inside RayPPOTrainer.__init__ to set self.use_critic (also threaded into
#   compute_data_metrics to toggle val/* keys).
# Called by: TaskRunner.run() / add_critic_worker in main_ppo.py,
#   RayPPOTrainer.__init__ in ray_trainer.py, main_ppo_sync.py.
# Branches:
#   - critic.enable explicitly set -> honor it (bool cast).
#   - adv_estimator == GAE -> True (PPO: bootstrap from V(s), needs a trained critic).
#   - GRPO / GDPO / REINFORCE++ / RLOO -> False, warn once: these use group- or
#     Monte-Carlo-baseline advantages, no learned value head.
# Why: Critic only exists when the advantage estimator bootstraps from a learned
#   value (GAE). GRPO and similar group-normalized estimators replace the baseline
#   with intra-group statistics, dropping the second trainable model entirely.
def need_critic(config: DictConfig) -> bool:
    """Given a config, do we need critic."""
    if config.critic.enable is not None:
        return bool(config.critic.enable)
    elif config.algorithm.adv_estimator == AdvantageEstimator.GAE:
        return True
    else:
        warnings.warn(
            "Disabled critic as algorithm.adv_estimator != gae. If it is not intended, please set critic.enable=True",
            stacklevel=2,
        )
        return False
