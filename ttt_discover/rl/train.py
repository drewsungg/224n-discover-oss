"""
Implements RL on general MDPs
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Literal, Sequence, cast

import chz
import numpy as np
import wandb
import math
import torch
from ttt_discover.opentinker_backend.data_types import TrainingDatum, TokenSequence
from ttt_discover.opentinker_backend.clients import (
    VLLMSamplingClient,
    OpenTinkerTrainingSession,
    datums_to_dataproto,
)
from ttt_discover.tinker_utils.misc_utils import get_last_checkpoint, save_checkpoint_async
from ttt_discover.tinker_utils.completers import TwoPhaseTokenCompleter
from ttt_discover.rl.data_processing import (
    assemble_training_data,
    remove_constant_reward_groups,
)
from ttt_discover.rl.metric_util import compute_trajectory_metrics
from ttt_discover.rl.rollouts import do_group_rollout
from ttt_discover.rl.types import (
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    TrajectoryGroup,
)
from ttt_discover.tinker_utils.misc_utils import Tokenizer
from ttt_discover.tinker_utils import ml_log
from ttt_discover.tinker_utils.misc_utils import safezip, split_list, timed, all_same
from ttt_discover.tinker_utils.trace import scope, get_scope_context
from ttt_discover.tinker_utils.ml_log import WandbLogger


logger = logging.getLogger(__name__)

# Loss function type (was tinker.types.LossFnType)
LossFnType = Literal["importance_sampling", "ppo"]


@scope
async def incorporate_kl_penalty(
    data_D: List[TrainingDatum],
    base_vllm_client: VLLMSamplingClient,
    kl_penalty_coef: float,
) -> Dict[str, float]:
    """
    Compute KL against base model. Adjust advantages in-place by logp_base - logp_current - avg_kl,
    where avg_kl is the average of logp_base - logp_current (which is -KL[current, base])
    """
    # Compute logprobs at all data items
    full_sequence_inputs_D = [
        datum.model_input.append_int(cast(int, datum.loss_fn_inputs["target_tokens"][-1].item()))
        for datum in data_D
    ]
    base_logprobs_D = await asyncio.gather(
        *[
            base_vllm_client.compute_logprobs_async(sequence_input)
            for sequence_input in full_sequence_inputs_D
        ]
    )
    # compute the logprob differences, zeroed out when the mask == 0
    sampled_logprobs_D = [datum.loss_fn_inputs["logprobs"] for datum in data_D]
    float_masks = [datum.loss_fn_inputs["mask"].float() for datum in data_D]
    logprob_diffs = [
        (sampled_logprobs - torch.tensor(base_logprobs[1:])) * mask
        for base_logprobs, sampled_logprobs, mask in safezip(
            base_logprobs_D, sampled_logprobs_D, float_masks
        )
    ]
    avg_logp_diff = sum([diff.sum() for diff in logprob_diffs]) / sum(
        [mask.sum() for mask in float_masks]
    )
    for i, datum in enumerate(data_D):
        kl_advantages = kl_penalty_coef * float_masks[i] * (avg_logp_diff - logprob_diffs[i])
        datum.loss_fn_inputs["advantages"] = (
            datum.loss_fn_inputs["advantages"] + kl_advantages
        )
    return {"kl_policy_base": float(avg_logp_diff)}


def compute_advantages(trajectory_groups_P: List[TrajectoryGroup], adv_estimator: str, adv_estimator_beta: float, adv_estimator_mu: float = 5/1.503163635, adv_estimator_sigma: float = 0.000001) -> List[torch.Tensor]:
    """Compute advantages for each trajectory, centered within groups."""
    advantages_P: list[torch.Tensor] = []

    for traj_group in trajectory_groups_P:
        rewards_G = torch.tensor(traj_group.get_total_rewards())
        # Center advantages within the group
        if adv_estimator == "mean_baseline":
            advantages_G = rewards_G - rewards_G.mean()
        elif adv_estimator == "entropic":
            beta = adv_estimator_beta
            s_safe = rewards_G - rewards_G.max(dim=-1, keepdim=True)[0]
            e = torch.exp(beta * s_safe)
            k = e.shape[0]
            if k == 1:
                Z = e
            else:
                Z = (e.sum() - e) / (k - 1)
            w = e / (Z + 1e-12)
            advantages_G = w - 1.0
        elif adv_estimator == "entropic_adaptive_beta":
            delta = np.log(2)
            beta_max = 1e6
            iters = 60
            eps = 1e-12

            r = rewards_G.float()
            k = r.shape[0]

            if k < 2:
                beta = r.new_tensor(0.0)
            else:
                logK = math.log(k)

                def kl_hat(beta_scalar: float) -> float:
                    # q_beta over samples: q ∝ exp(beta * r), KL(q||uniform)
                    b = r.new_tensor(beta_scalar)
                    logits = b * (r - r.max(dim=0, keepdim=True).values)      # stable
                    logq = logits - torch.logsumexp(logits, dim=0, keepdim=True)
                    q = torch.exp(logq)
                    kl = (q * (logq + logK)).sum(dim=0)
                    return float(kl.mean().item())

                lo, hi = 0.0, 1.0
                if kl_hat(hi) < delta:
                    while hi < beta_max and kl_hat(hi) < delta:
                        hi *= 2.0
                    if kl_hat(hi) < delta:
                        beta = r.new_tensor(hi)  # best effort
                    else:
                        beta = None
                else:
                    beta = None

                if beta is None:
                    for _ in range(iters):
                        mid = 0.5 * (lo + hi)
                        if kl_hat(mid) < delta:
                            lo = mid
                        else:
                            hi = mid
                    beta = r.new_tensor(hi)

            # LOO entropic advantages using solved beta
            e = torch.exp(beta * (r - r.max(dim=0, keepdim=True).values))

            if k == 1:
                Z = e
            else:
                Z = (e.sum(dim=0, keepdim=True) - e) / (k - 1)

            w = e / (Z + eps)
            advantages_G = w - 1.0
        else:
            raise ValueError(f"Invalid advantage estimator: {adv_estimator}")
        advantages_P.append(advantages_G)

    return advantages_P


@scope
async def train_step(
    data_D: List[TrainingDatum],
    training_session: OpenTinkerTrainingSession,
    learning_rate: float,
    num_substeps: int,
    loss_fn: LossFnType,
    tokenizer: Tokenizer,
) -> List[torch.Tensor]:
    """Train the model on collected trajectories using OpenTinker's train_step API."""
    if len(data_D) == 0:
        return []

    # Convert TrainingDatums to DataProto for OpenTinker
    batch = datums_to_dataproto(data_D, tokenizer)

    # Execute training step via OpenTinker server
    result = await asyncio.to_thread(training_session.train_step, batch)

    # Extract training logprobs from result if available
    training_logprobs_D: list[torch.Tensor] = []
    metrics = result.get("metrics", {})
    if metrics:
        logger.info(f"Train step metrics: {metrics}")

    return training_logprobs_D


@chz.chz
class Config:
    env_type: type  # Environment type (from registry or passed directly for custom envs)
    problem_type: str
    learning_rate: float
    dataset_builder: RLDatasetBuilder  # also determines batch size
    model_name: str
    num_epochs: int = 1
    temperature: float = 1.0  # Changing sampling temperature is not generally recommended; does not currently play well with KL penalty
    lora_rank: int = 32
    adv_estimator: str="entropic_adaptive_beta"
    adv_estimator_beta: float = 2.0

    kl_penalty_coef: float = 0.0

    # Loss function to use for training: "importance_sampling" or "ppo"
    loss_fn: LossFnType = "importance_sampling"

    # Number of optimizer steps per training iteration.
    # Useful for very large batch sizes.
    num_substeps: int = 1

    wandb_project: str | None = None
    wandb_name: str | None = None

    log_path: str = chz.field(munger=lambda _, s: os.path.expanduser(s))
    enable_trace: bool = False

    remove_constant_reward_groups: bool = False
    save_every: int = 20  # 0 = disabled
    load_checkpoint_path: str | None = None

    # Two-phase sampling: phase1_max_tokens for token completion
    phase1_max_tokens: int = 26000

    # Local model path (avoids HuggingFace API rate limits)
    local_model_path: str | None = None

    # OpenTinker server configuration
    server_url: str | None = None
    scheduler_url: str | None = None
    vllm_server_url: str | None = None
    num_gpus: int | None = None


@chz.chz
class WrappedTrajectoryGroup:
    """
    A wrapper around a trajectory group that includes metadata about how it was generated.
    Used when we need to overlap sampling and training.
    """

    trajectory_group: TrajectoryGroup
    # The env group builder that produced the trajectory group.
    # Pass this along in case the sampler is too stale, and we need to
    # requeue this group.
    env_group_builder: EnvGroupBuilder
    # The step that produced this trajectory group.
    sampling_client_step: int
    metrics: dict[str, Any] = chz.field(default_factory=dict)


@scope
async def do_group_rollout_and_filter_constant_reward(
    vllm_client: VLLMSamplingClient,
    env_group_builder: EnvGroupBuilder,
    temperature: float,
    do_remove_constant_reward_groups: bool,
    step_idx=-1,
    model_name: str = "",
    phase1_max_tokens: int = 27000,
) -> TrajectoryGroup | None:
    from ttt_discover.tinker_utils.misc_utils import get_tokenizer

    tokenizer = get_tokenizer(model_name)

    policy = TwoPhaseTokenCompleter(
        sampling_client=vllm_client,
        tokenizer=tokenizer,
        phase1_max_tokens=phase1_max_tokens,
        temperature=temperature,
    )

    trajectory_group = await do_group_rollout(env_group_builder, policy, step_idx)

    # Remove if all trajectories have the same reward
    if do_remove_constant_reward_groups and all_same(trajectory_group.get_total_rewards()):
        return None
    else:
        return trajectory_group


@scope
async def save_checkpoint_and_get_vllm_client(
    training_session: OpenTinkerTrainingSession,
    vllm_client: VLLMSamplingClient,
    i_batch: int,
    log_path: str,
    save_every: int,
    start_batch: int = 0,
) -> tuple[VLLMSamplingClient, dict[str, Any]]:
    metrics = {}
    with timed("save_checkpoint", metrics):
        if save_every > 0 and i_batch > start_batch and i_batch % save_every == 0:
            await save_checkpoint_async(
                training_session=training_session,
                name=f"{i_batch:06d}",
                log_path=log_path,
                loop_state={"batch": i_batch},
                kind="both",
            )
    # Return the same vLLM client — the server's model weights are updated in-place
    # after each train_step, so the vLLM client will use the latest weights
    # when it's pointed at the training server's inference endpoint.
    return vllm_client, metrics


@scope
async def prepare_minibatch(
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
    tokenizer: Tokenizer,
    vllm_base_client: VLLMSamplingClient | None,
    model_name: str,
    kl_penalty_coef: float,
    log_path: str | None = None,
    train_step: int | None = None,
    adv_estimator: str="mean_baseline",
    adv_estimator_beta: float = 2.0,
) -> tuple[list[TrainingDatum], dict[str, Any]]:
    """Converts the trajectories into a minibatch, and provides metrics about the minibatch"""

    # Compute trajectory metrics
    metrics = {}
    taglist_P = [env_group_builder.logging_tags() for env_group_builder in env_group_builders_P]
    metrics.update(compute_trajectory_metrics(trajectory_groups_P, taglist_P))

    # Assemble training data
    with timed("assemble_training_data", metrics):
        advantages_P = compute_advantages(trajectory_groups_P, adv_estimator, adv_estimator_beta=adv_estimator_beta)
        if advantages_P:
            flat_adv = torch.cat(advantages_P)
            metrics.update(
                {
                    "advantage/mean": flat_adv.mean().item(),
                    "advantage/min": flat_adv.min().item(),
                    "advantage/max": flat_adv.max().item(),
                }
            )
        data_D, _metadata_D = assemble_training_data(trajectory_groups_P, advantages_P)

    # Incorporate KL penalty if configured
    if kl_penalty_coef > 0 and vllm_base_client is not None:
        with timed("kl_vs_base", metrics):
            kl_penalty_metrics = await incorporate_kl_penalty(
                data_D,
                vllm_base_client,
                kl_penalty_coef,
            )
        metrics.update(kl_penalty_metrics)

    return data_D, metrics


@scope
async def do_train_step_and_get_vllm_client(
    cfg: Config,
    i_batch: int,
    training_session: OpenTinkerTrainingSession,
    vllm_client: VLLMSamplingClient,
    vllm_base_client: VLLMSamplingClient | None,
    tokenizer: Tokenizer,
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
) -> tuple[VLLMSamplingClient, dict[str, Any]]:
    context = get_scope_context()
    context.attributes["step"] = i_batch

    metrics = {}
    data_D, prepare_minibatch_metrics = await prepare_minibatch(
        env_group_builders_P,
        trajectory_groups_P,
        tokenizer,
        vllm_base_client,
        model_name=cfg.model_name,
        kl_penalty_coef=cfg.kl_penalty_coef,
        log_path=cfg.log_path,
        train_step=i_batch,
        adv_estimator=cfg.adv_estimator,
        adv_estimator_beta=cfg.adv_estimator_beta,
    )
    metrics.update(prepare_minibatch_metrics)

    with timed("train", metrics):
        training_logprobs_D = await train_step(
            data_D,
            training_session,
            cfg.learning_rate,
            cfg.num_substeps,
            cfg.loss_fn,
            tokenizer,
        )

    vllm_client, full_batch_metrics = await save_checkpoint_and_get_vllm_client(
        training_session,
        vllm_client,
        # NOTE: saving the checkpoint as the i + 1 step
        i_batch + 1,
        cfg.log_path,
        cfg.save_every,
    )
    metrics.update(full_batch_metrics)

    return vllm_client, metrics


@scope
async def do_sync_training(
    start_batch: int,
    end_batch: int,
    num_batches: int,
    cfg: Config,
    training_session: OpenTinkerTrainingSession,
    vllm_client: VLLMSamplingClient,
    vllm_base_client: VLLMSamplingClient | None,
    dataset: RLDataset,
    ml_logger: ml_log.Logger,
    tokenizer: Tokenizer,
):
    """Implements fully synchronous on-policy training"""
    num_batches_per_epoch = len(dataset)
    if num_batches_per_epoch == 0:
        raise ValueError("RLDataset must contain at least one batch")

    for i_batch in range(start_batch, end_batch):
        train_table = None
        test_table = None
        metrics = {
            "progress/batch": i_batch,
            "optim/lr": cfg.learning_rate,
            "progress/done_frac": (i_batch + 1) / num_batches,
        }
        t_start = time.time()

        # Make sure we are clearing out existing entrees
        # in case of resuming from previous checkpoints
        from ttt_discover.tinker_utils.best_sequence_utils import get_best_bound_path, clear_step_entry
        best_seq_path = get_best_bound_path(cfg.log_path)
        clear_step_entry(best_seq_path, i_batch)

        # Get batch and sample trajectories
        print("Load dataset batch...")
        dataset_batch_idx = i_batch % num_batches_per_epoch
        env_group_builders_P = dataset.get_batch(dataset_batch_idx)

        # Log sampler stats if available (PER sampler)
        print("Log sampler stats...")
        sampler_table_columns, sampler_table_data = None, None
        if hasattr(dataset, 'sampler') and hasattr(dataset.sampler, 'get_sample_stats'):
            sampler_stats = dataset.sampler.get_sample_stats()
            metrics.update(sampler_stats)
            if hasattr(dataset.sampler, 'get_sample_table'):
                sampler_table_columns, sampler_table_data = dataset.sampler.get_sample_table()


        print("Sampling...")
        with timed("sampling", metrics):
            # Note: do_remove_constant_reward_groups=False here because we remove
            # constant reward groups after all rollouts are collected (below)
            trajectory_groups_P = await asyncio.gather(
                *[
                    asyncio.create_task(
                        do_group_rollout_and_filter_constant_reward(
                            vllm_client,
                            builder,
                            temperature=cfg.temperature,
                            do_remove_constant_reward_groups=False,
                            step_idx=i_batch,
                            model_name=cfg.local_model_path or cfg.model_name,
                            phase1_max_tokens=cfg.phase1_max_tokens,
                        ),
                        name=f"sample_task_{i}",
                    )
                    for i, builder in enumerate(env_group_builders_P)
                ],
            )

        if hasattr(dataset, 'flush'):
            dataset.flush(step=i_batch + 1)

        if cfg.remove_constant_reward_groups:
            trajectory_groups_P = remove_constant_reward_groups(trajectory_groups_P)

        # Train step
        print("Training...")
        vllm_client, train_step_metrics = await do_train_step_and_get_vllm_client(
            cfg,
            i_batch,
            training_session,
            vllm_client,
            vllm_base_client,
            tokenizer,
            env_group_builders_P,
            trajectory_groups_P,
        )

        if 'table' in train_step_metrics:
            table_data = train_step_metrics.pop('table')
            # Compute actual advantages using the configured estimator
            advantages_P = compute_advantages(trajectory_groups_P, cfg.adv_estimator, cfg.adv_estimator_beta)
            flat_advantages = [adv.item() for adv_G in advantages_P for adv in adv_G]
            table_data = [(*row, flat_advantages[i]) for i, row in enumerate(table_data)]
            train_table = {
                f"gen&score_train_{i_batch}":
                    wandb.Table(
                        columns=[
                            "Prompt", "Gen Sequence", "Reward", "Correctness", "Gen Sequence PostProc", "Message", "Initial Raw Score", "Advantage"
                        ],
                        data=table_data
                    )
            }

        if len(ml_logger.loggers) >= 2:
            if train_table is not None and isinstance(ml_logger.loggers[2], WandbLogger):
                ml_logger.loggers[2].log_metrics(train_table, step=i_batch)
            if test_table is not None and isinstance(ml_logger.loggers[2], WandbLogger):
                ml_logger.loggers[2].log_metrics(test_table, step=i_batch)
            if sampler_table_data is not None and isinstance(ml_logger.loggers[2], WandbLogger):
                ml_logger.loggers[2].log_metrics({
                    f"sampler_states_{i_batch}": wandb.Table(
                        columns=sampler_table_columns,
                        data=sampler_table_data
                    )
                }, step=i_batch)

        # Log metrics
        metrics.update(train_step_metrics)
        metrics["time/total"] = time.time() - t_start
        ml_logger.log_metrics(metrics, step=i_batch)


@scope
async def main(
    cfg: Config,
):
    """Main training loop for MDP RL."""
    if cfg.num_epochs < 1:
        raise ValueError("num_epochs must be >= 1")

    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        wandb_project=cfg.wandb_project,
        config=cfg,
        wandb_name=cfg.wandb_name,
    )

    resume_info = get_last_checkpoint(cfg.log_path)
    if resume_info:
        start_batch = resume_info["batch"]
    else:
        start_batch = 0

    # Create OpenTinker training session
    print("Creating OpenTinker training session...")
    training_config = {
        "model_name": cfg.model_name,
        "learning_rate": cfg.learning_rate,
        "lora_rank": cfg.lora_rank,
        "loss_fn": cfg.loss_fn,
    }
    if cfg.load_checkpoint_path:
        training_config["checkpoint_path"] = cfg.load_checkpoint_path
    if resume_info:
        training_config["resume_path"] = resume_info.get("state_path")

    training_session = OpenTinkerTrainingSession(
        server_url=cfg.server_url,
        scheduler_url=cfg.scheduler_url,
        config=training_config,
        num_gpus=cfg.num_gpus,
    )
    print("Training session created!")

    # Create vLLM sampling client for inference
    vllm_url = cfg.vllm_server_url
    if not vllm_url:
        raise ValueError("vllm_server_url must be provided for sampling")
    vllm_client = VLLMSamplingClient(
        vllm_server_url=vllm_url,
        model_name=cfg.model_name,
    )

    # Create base model vLLM client for KL penalty (if needed)
    vllm_base_client: VLLMSamplingClient | None = None
    if cfg.kl_penalty_coef > 0:
        # Use a separate vLLM server for the base model
        # TODO: make base_vllm_server_url configurable
        vllm_base_client = VLLMSamplingClient(
            vllm_server_url=vllm_url,
            model_name=cfg.model_name,
        )

    # Get tokenizer
    if cfg.local_model_path:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.local_model_path, use_fast=True)
    else:
        from ttt_discover.tinker_utils.misc_utils import get_tokenizer
        tokenizer = get_tokenizer(cfg.model_name)

    # Create dataset from thunk
    print("Create dataset...")
    dataset = await cfg.dataset_builder()
    print("Dataset created!")

    # If resuming from step > 0, reload sampler from the correct checkpoint step
    if resume_info and start_batch > 0 and hasattr(dataset, 'sampler') and hasattr(dataset.sampler, 'reload_from_step'):
        logger.info(f"Reloading sampler state from step {start_batch}")
        dataset.sampler.reload_from_step(start_batch)

    num_batches_per_epoch = len(dataset)
    if num_batches_per_epoch == 0:
        raise ValueError("RLDataset must contain at least one batch")
    num_batches_total = num_batches_per_epoch * cfg.num_epochs
    logger.info(
        f"Will train for {cfg.num_epochs} epoch(s) x {num_batches_per_epoch} batches = {num_batches_total} steps"
    )

    # Training loop
    print("Training loop...")
    await do_sync_training(
        start_batch=start_batch,
        end_batch=num_batches_total,
        num_batches=num_batches_total,
        cfg=cfg,
        training_session=training_session,
        vllm_client=vllm_client,
        vllm_base_client=vllm_base_client,
        dataset=dataset,
        ml_logger=ml_logger,
        tokenizer=tokenizer,
    )

    # Save final checkpoint
    if start_batch < num_batches_total:
        await save_checkpoint_async(
            training_session=training_session,
            name="final",
            log_path=cfg.log_path,
            kind="both",
            loop_state={"batch": num_batches_total},
        )
    else:
        logger.info("Training was already complete; nothing to do")

    # Cleanup
    await vllm_client.close()
    if vllm_base_client:
        await vllm_base_client.close()
    ml_logger.close()
    logger.info("Training completed successfully")
