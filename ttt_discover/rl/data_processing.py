"""
Data processing functions for RL training.

Contains functions for computing advantages, converting trajectories to training data,
and assembling training batches.
"""

import logging
from typing import List

import torch
from ttt_discover.opentinker_backend.data_types import TokenSequence, TrainingDatum
from ttt_discover.rl.types import Trajectory, TrajectoryGroup
from ttt_discover.tinker_utils.misc_utils import all_same, safezip

logger = logging.getLogger(__name__)


def create_rightshifted_model_input_and_leftshifted_targets(
    tokens: list[int],
) -> tuple[TokenSequence, list[int]]:
    """
    Given a full sequence of tokens, create
     "inputs" (with last token removed)
     "targets" (with first token removed)
    """
    if len(tokens) < 2:
        raise ValueError("need at least 2 tokens for input/target split")

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]

    return TokenSequence(tokens=input_tokens), target_tokens


def _is_prefix(seq1: list[int], seq2: list[int]) -> bool:
    """
    Check if seq1 is a prefix of seq2.
    """
    return len(seq1) <= len(seq2) and seq2[: len(seq1)] == seq1


def trajectory_to_data(traj: Trajectory, traj_advantage: float) -> list[TrainingDatum]:
    """
    Return one or more TrainingDatum objects corresponding to the trajectory.
    If the sequence grows by appending, i.e., each successive observation contains
    the previous observation+action as a prefix, then we can return a single TrainingDatum.
    However, if we get a sequence that's not an extension of the previous sequence,
    then that results in a new TrainingDatum.

    For example, let O1 denote a chunk of observation tokens, and let A1 denote an action.

    Then let's say ob_ac_pairs is as follows.

    (O1, A1)
    (O1+A1+O2, A2)
    (O3, A3)

    Then we will merge the first two observation-action pairs into a single TrainingDatum,
    and the last observation-action pair into a separate TrainingDatum.
    """

    class SequenceAccumulator:
        full_sequence: list[int] = []
        sampled_logprobs: list[float] = []
        advantages: list[float] = []
        mask: list[float] = []

        @classmethod
        def clear(cls):
            cls.full_sequence = []
            cls.sampled_logprobs = []
            cls.advantages = []
            cls.mask = []

    def make_datum_from_state():
        all_tokens = SequenceAccumulator.full_sequence
        input_tokens_T, target_tokens_T = create_rightshifted_model_input_and_leftshifted_targets(
            all_tokens
        )
        sampled_logprobs_T = SequenceAccumulator.sampled_logprobs[1:]
        advantages_T = SequenceAccumulator.advantages[1:]
        mask_T = SequenceAccumulator.mask[1:]
        assert (
            input_tokens_T.length
            == len(target_tokens_T)
            == len(sampled_logprobs_T)
            == len(advantages_T)
            == len(mask_T)
        )
        return TrainingDatum(
            model_input=input_tokens_T,
            loss_fn_inputs={
                "target_tokens": torch.tensor(target_tokens_T),
                "logprobs": torch.tensor(sampled_logprobs_T),
                "advantages": torch.tensor(advantages_T),
                "mask": torch.tensor(mask_T),
            },
        )

    data: list[TrainingDatum] = []
    for transition in traj.transitions:
        ob = transition.ob
        ob_flat = list(ob.tokens)
        ac_with_logprobs = transition.ac
        if len(SequenceAccumulator.full_sequence) == 0:
            delta_ob_flat = ob_flat
        elif _is_prefix(SequenceAccumulator.full_sequence, ob_flat):
            delta_ob_flat = ob_flat[len(SequenceAccumulator.full_sequence) :]
        else:
            data.append(make_datum_from_state())
            SequenceAccumulator.clear()
            delta_ob_flat = ob_flat
        delta_ob_len = len(delta_ob_flat)
        SequenceAccumulator.full_sequence.extend(delta_ob_flat)
        SequenceAccumulator.full_sequence.extend(ac_with_logprobs.tokens)
        SequenceAccumulator.sampled_logprobs.extend(
            [0.0] * delta_ob_len + ac_with_logprobs.logprobs
        )
        SequenceAccumulator.advantages.extend(
            [0] * delta_ob_len + [traj_advantage] * len(ac_with_logprobs.tokens)
        )
        SequenceAccumulator.mask.extend([0.0] * delta_ob_len + ac_with_logprobs.mask)

    if SequenceAccumulator.full_sequence:
        data.append(make_datum_from_state())

    return data


def assemble_training_data(
    trajectory_groups_P: List[TrajectoryGroup],
    advantages_P: List[torch.Tensor],
) -> tuple[List[TrainingDatum], List[dict[str, int]]]:
    """Convert trajectories to training data format."""
    data_D: list[TrainingDatum] = []
    metadata_D: list[dict[str, int]] = []

    for i_group, (traj_group, advantages_G) in enumerate(
        safezip(trajectory_groups_P, advantages_P)
    ):
        for i_traj, (traj, traj_advantage) in enumerate(
            safezip(traj_group.trajectories_G, advantages_G)
        ):
            # Build the full sequence from the trajectory
            new_data = trajectory_to_data(traj, float(traj_advantage))
            data_D.extend(new_data)
            metadata_D.extend([dict(group_idx=i_group, traj_idx=i_traj) for _ in new_data])

    return data_D, metadata_D


def remove_constant_reward_groups(
    trajectory_groups_P: List[TrajectoryGroup],
) -> List[TrajectoryGroup]:
    new_groups: list[TrajectoryGroup] = []
    for group in trajectory_groups_P:
        if not all_same(group.get_total_rewards()):
            new_groups.append(group)
    if not new_groups:
        logger.warning("All rewards are uniform. There will be no gradient")
        return trajectory_groups_P[0:1]  # return singleton list in case empty
        # list will cause problems
    return new_groups
