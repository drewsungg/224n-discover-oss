"""
Client wrappers for OpenTinker's training and inference APIs.

VLLMSamplingClient replaces tinker.SamplingClient.
OpenTinkerTrainingSession replaces tinker.ServiceClient + tinker.TrainingClient.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

import aiohttp
import torch

from ttt_discover.opentinker_backend.data_types import (
    SampleResult,
    SampleSequence,
    SamplingParams,
    TokenSequence,
    TrainingDatum,
)

logger = logging.getLogger(__name__)


class VLLMSamplingClient:
    """Token-level sampling client using vLLM's OpenAI-compatible API.

    Replaces tinker.SamplingClient with equivalent functionality:
    - sample_async: generate tokens with per-token logprobs
    - compute_logprobs_async: compute logprobs for an existing sequence
    """

    def __init__(
        self,
        vllm_server_url: str,
        model_name: str | None = None,
    ):
        self.vllm_server_url = vllm_server_url.rstrip("/")
        self.model_name = model_name or "default"
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def sample_async(
        self,
        prompt: TokenSequence,
        num_samples: int,
        sampling_params: SamplingParams,
    ) -> SampleResult:
        """Generate token completions with per-token logprobs.

        Uses vLLM's /v1/completions endpoint with prompt_token_ids.
        """
        session = await self._get_session()

        # Build stop parameter
        stop: list[str] | None = None
        if sampling_params.stop and isinstance(sampling_params.stop[0], str):
            stop = sampling_params.stop

        request_body: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt.tokens,
            "max_tokens": sampling_params.max_tokens,
            "temperature": sampling_params.temperature,
            "n": num_samples,
            "logprobs": 1,
        }
        if stop:
            request_body["stop"] = stop
        # If stop tokens are ints, use stop_token_ids
        if sampling_params.stop and isinstance(sampling_params.stop[0], int):
            request_body["stop_token_ids"] = sampling_params.stop

        url = f"{self.vllm_server_url}/v1/completions"
        async with session.post(url, json=request_body) as resp:
            resp.raise_for_status()
            result = await resp.json()

        sequences = []
        for choice in result["choices"]:
            tokens = choice.get("logprobs", {}).get("tokens", [])
            token_ids = choice.get("logprobs", {}).get("token_ids", [])
            token_logprobs = choice.get("logprobs", {}).get("token_logprobs", [])

            # If token_ids not directly available, use the text tokens
            # vLLM returns token_logprobs as a list of floats
            if not token_ids and "text" in choice:
                # Fallback: we can't easily get token IDs from text in all cases.
                # Most vLLM versions return token_ids in logprobs.
                token_ids = tokens  # Will need tokenizer to convert

            logprobs = [lp if lp is not None else 0.0 for lp in token_logprobs]
            sequences.append(SampleSequence(tokens=token_ids, logprobs=logprobs))

        return SampleResult(sequences=sequences)

    async def compute_logprobs_async(
        self, sequence: TokenSequence
    ) -> list[float]:
        """Compute log probabilities for an existing token sequence.

        Uses vLLM's /v1/completions with prompt_logprobs and max_tokens=0
        to score the sequence without generating new tokens.
        """
        session = await self._get_session()

        request_body: dict[str, Any] = {
            "model": self.model_name,
            "prompt": sequence.tokens,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 1,
            "echo": True,
        }

        url = f"{self.vllm_server_url}/v1/completions"
        async with session.post(url, json=request_body) as resp:
            resp.raise_for_status()
            result = await resp.json()

        # Extract prompt logprobs from the echo response
        choice = result["choices"][0]
        token_logprobs = choice.get("logprobs", {}).get("token_logprobs", [])

        # First token logprob is typically None (no conditioning), replace with 0
        return [lp if lp is not None else 0.0 for lp in token_logprobs]

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


class OpenTinkerTrainingSession:
    """Wraps OpenTinker's HTTPTrainingClient + SchedulerClient.

    Replaces tinker.ServiceClient + tinker.TrainingClient with a unified
    session that manages the training server lifecycle.
    """

    def __init__(
        self,
        server_url: str | None = None,
        scheduler_url: str | None = None,
        config: dict[str, Any] | None = None,
        api_key: str | None = None,
        num_gpus: int | None = None,
    ):
        # Import here to avoid hard dependency at module level
        import sys
        sys.path.insert(0, "/home/ubuntu/224n-discover-oss/OpenTinker")
        from opentinker.client.utils.http_training_client import (
            HTTPTrainingClient,
            SchedulerClient,
        )

        self.job_id: str | None = None

        if server_url:
            # Direct connection to existing server
            self.client = HTTPTrainingClient(server_url)
        elif scheduler_url:
            # Submit job to scheduler, get server URL
            self.scheduler_client = SchedulerClient(scheduler_url, api_key=api_key)
            job_result = self.scheduler_client.submit_job(
                config=config or {},
                enable_agent_loop=True,
                num_gpus=num_gpus,
            )
            self.job_id = job_result["job_id"]
            self.client = HTTPTrainingClient(job_result["server_url"])
        else:
            raise ValueError("Either server_url or scheduler_url must be provided")

    def init_workers(self, total_steps: int = 100, timeout: float = 600.0) -> dict:
        return self.client.init_workers(total_steps=total_steps, timeout=timeout)

    def set_config(self, config: dict[str, Any], env=None) -> dict:
        return self.client.set_config(config, env=env)

    def set_generation_config(self, config: dict[str, Any]) -> dict:
        return self.client.set_generation_config(config)

    def train_step(self, batch) -> dict:
        """Execute one training step. batch should be a verl DataProto."""
        return self.client.train_step(batch)

    def validate(self, batch) -> dict:
        return self.client.validate(batch)

    def save_checkpoint(self) -> dict:
        return self.client.save_checkpoint()

    def upload_reward_function(self, function_name: str, source_code: str) -> dict:
        return self.client.upload_reward_function(function_name, source_code)

    def health_check(self) -> dict:
        return self.client.health_check()

    def get_status(self) -> dict:
        return self.client.get_status()


def datums_to_dataproto(datums: list[TrainingDatum], tokenizer) -> Any:
    """Convert TTT-Discover's TrainingDatums to OpenTinker's verl DataProto.

    Pads sequences to uniform length, builds attention masks, and packs
    advantages/logprobs into the batch format expected by OpenTinker's
    training server.
    """
    import sys
    sys.path.insert(0, "/home/ubuntu/224n-discover-oss/OpenTinker")
    from verl import DataProto
    from tensordict import TensorDict

    if not datums:
        raise ValueError("Cannot convert empty datum list to DataProto")

    # Find max sequence lengths
    max_input_len = max(d.model_input.length for d in datums)
    max_target_len = max(d.loss_fn_inputs["target_tokens"].shape[0] for d in datums)
    batch_size = len(datums)

    # Build padded tensors
    input_ids = torch.zeros(batch_size, max_input_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_input_len, dtype=torch.long)
    target_tokens = torch.zeros(batch_size, max_target_len, dtype=torch.long)
    logprobs = torch.zeros(batch_size, max_target_len, dtype=torch.float)
    advantages = torch.zeros(batch_size, max_target_len, dtype=torch.float)
    mask = torch.zeros(batch_size, max_target_len, dtype=torch.float)

    for i, datum in enumerate(datums):
        # Input tokens
        tokens = datum.model_input.tokens
        seq_len = len(tokens)
        input_ids[i, :seq_len] = torch.tensor(tokens, dtype=torch.long)
        attention_mask[i, :seq_len] = 1

        # Loss function inputs
        tgt_len = datum.loss_fn_inputs["target_tokens"].shape[0]
        target_tokens[i, :tgt_len] = datum.loss_fn_inputs["target_tokens"]
        logprobs[i, :tgt_len] = datum.loss_fn_inputs["logprobs"]
        advantages[i, :tgt_len] = datum.loss_fn_inputs["advantages"]
        mask[i, :tgt_len] = datum.loss_fn_inputs["mask"]

    batch = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "responses": target_tokens,
            "old_log_probs": logprobs,
            "advantages": advantages,
            "response_mask": mask,
        },
        batch_size=batch_size,
    )

    return DataProto(batch=batch)
