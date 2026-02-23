"""
Implementations that correspond to a model or policy that can be sampled from, but with different amounts of additional structure.

The TokenCompleter operates on tokens. This is the version used by RL algorithms, because RL algorithms work on Tokens. The MessageCompleter operates on messages, so it needs to be used with a renderer.

Evals and other code should use the appropriate interface.
"""

from dataclasses import dataclass
from typing import TypeAlias

from ttt_discover.opentinker_backend.data_types import SamplingParams, TokenSequence
from ttt_discover.opentinker_backend.clients import VLLMSamplingClient
from ttt_discover.tinker_utils.misc_utils import Tokenizer

# Interfaces

StopCondition: TypeAlias = list[str] | list[int]


@dataclass
class TokensWithLogprobs:
    tokens: list[int]
    maybe_logprobs: list[float] | None
    maybe_mask: list[float] | None = None  # Optional mask: 1.0 = train, 0.0 = don't train

    @property
    def logprobs(self) -> list[float]:
        if self.maybe_logprobs is None:
            raise ValueError("Logprobs are not available")
        return self.maybe_logprobs

    @property
    def mask(self) -> list[float]:
        """Return mask, defaulting to all 1.0 if not provided."""
        if self.maybe_mask is None:
            return [1.0] * len(self.tokens)
        return self.maybe_mask


class TokenCompleter:
    async def __call__(
        self, model_input: TokenSequence, stop: StopCondition
    ) -> TokensWithLogprobs:
        raise NotImplementedError


@dataclass
class TwoPhaseTokenCompleter(TokenCompleter):
    """
    Two-phase completer for gpt-oss: if Phase 1 exhausts tokens without stop, Phase 2 forces final answer.
    Uses full context window dynamically.
    """
    sampling_client: VLLMSamplingClient
    tokenizer: Tokenizer
    phase1_max_tokens: int  # Phase 1 limit (e.g., 27000)
    temperature: float = 1.0
    context_window: int = 32768
    context_buffer: int = 50

    PHASE2_PREFILL = "\n\n... okay, I am out of thinking tokens. I need to send my final message now."
    # Full marker to transition from analysis to final channel
    GPTOSS_FINAL_MARKER = "<|end|><|start|>assistant<|channel|>final<|message|>"
    # Marker that indicates we're already in the final channel
    GPTOSS_FINAL_CHANNEL_INDICATOR = "<|channel|>final<|message|>"

    def _hit_stop_sequence(self, tokens: list[int], stop: StopCondition) -> bool:
        """Check if the last token(s) match any stop sequence."""
        if not tokens:
            return False
        for s in stop:
            if isinstance(s, int):
                if tokens[-1] == s:
                    return True
            else:
                stop_tokens = self.tokenizer.encode(s, add_special_tokens=False)
                if len(stop_tokens) <= len(tokens) and tokens[-len(stop_tokens):] == stop_tokens:
                    return True
        return False

    def _contains_subsequence(self, tokens: list[int], pattern: str) -> bool:
        """Check if tokens contain the given pattern as a subsequence."""
        pattern_tokens = self.tokenizer.encode(pattern, add_special_tokens=False)
        if len(pattern_tokens) > len(tokens):
            return False
        for i in range(len(tokens) - len(pattern_tokens) + 1):
            if tokens[i:i + len(pattern_tokens)] == pattern_tokens:
                return True
        return False

    async def __call__(self, model_input: TokenSequence, stop: StopCondition) -> TokensWithLogprobs:
        prompt_length = model_input.length

        # phase1_max_tokens is the total context budget for phase 1 (prompt + output)
        # This guarantees (context_window - phase1_max_tokens - buffer) tokens for phase 2
        phase1_max = self.phase1_max_tokens - prompt_length
        if phase1_max <= 0:
            raise ValueError(f"Prompt length {prompt_length} exceeds phase1_max_tokens {self.phase1_max_tokens}.")

        phase1_result = await self.sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=SamplingParams(stop=stop, max_tokens=phase1_max, temperature=self.temperature),
        )
        phase1_tokens = phase1_result.sequences[0].tokens
        phase1_logprobs = phase1_result.sequences[0].logprobs
        assert phase1_logprobs is not None

        # Check if we hit stop sequence
        if self._hit_stop_sequence(phase1_tokens, stop) or len(phase1_tokens) < phase1_max:
            return TokensWithLogprobs(tokens=phase1_tokens, maybe_logprobs=phase1_logprobs)

        # Phase 2: Didn't hit stop, force completion
        # Phase 2 budget = context_window - prompt - phase1 - buffer

        # Already in final channel? Just continue without prefill
        if self._contains_subsequence(phase1_tokens, self.GPTOSS_FINAL_CHANNEL_INDICATOR):
            phase2_prompt = TokenSequence(tokens=model_input.tokens + phase1_tokens)
            phase2_max = self.context_window - prompt_length - len(phase1_tokens) - self.context_buffer
            if phase2_max <= 0:
                return TokensWithLogprobs(tokens=phase1_tokens, maybe_logprobs=phase1_logprobs)
            phase2_result = await self.sampling_client.sample_async(
                prompt=phase2_prompt, num_samples=1,
                sampling_params=SamplingParams(stop=stop, max_tokens=phase2_max, temperature=self.temperature),
            )
            phase2_tokens = phase2_result.sequences[0].tokens
            phase2_logprobs = phase2_result.sequences[0].logprobs
            assert phase2_logprobs is not None
            return TokensWithLogprobs(tokens=phase1_tokens + phase2_tokens, maybe_logprobs=phase1_logprobs + phase2_logprobs)

        # Need prefill to transition to final channel
        end_token_seq = self.tokenizer.encode("<|end|>", add_special_tokens=False)
        ends_with_end = len(end_token_seq) <= len(phase1_tokens) and phase1_tokens[-len(end_token_seq):] == end_token_seq
        if ends_with_end:
            prefill_text = self.PHASE2_PREFILL + "<|start|>assistant<|channel|>final<|message|>"
        else:
            prefill_text = self.PHASE2_PREFILL + self.GPTOSS_FINAL_MARKER
        prefill_tokens = self.tokenizer.encode(prefill_text, add_special_tokens=False)

        phase2_prompt = TokenSequence(tokens=model_input.tokens + phase1_tokens + prefill_tokens)
        phase2_max = self.context_window - prompt_length - len(phase1_tokens) - len(prefill_tokens) - self.context_buffer
        if phase2_max <= 0:
            return TokensWithLogprobs(
                tokens=phase1_tokens + prefill_tokens,
                maybe_logprobs=phase1_logprobs + [0.0] * len(prefill_tokens),
                maybe_mask=[1.0] * len(phase1_tokens) + [0.0] * len(prefill_tokens),
            )

        phase2_result = await self.sampling_client.sample_async(
            prompt=phase2_prompt, num_samples=1,
            sampling_params=SamplingParams(stop=stop, max_tokens=phase2_max, temperature=self.temperature),
        )
        phase2_tokens = phase2_result.sequences[0].tokens
        phase2_logprobs = phase2_result.sequences[0].logprobs
        assert phase2_logprobs is not None

        return TokensWithLogprobs(
            tokens=phase1_tokens + prefill_tokens + phase2_tokens,
            maybe_logprobs=phase1_logprobs + [0.0] * len(prefill_tokens) + phase2_logprobs,
            maybe_mask=[1.0] * len(phase1_tokens) + [0.0] * len(prefill_tokens) + [1.0] * len(phase2_tokens),
        )
