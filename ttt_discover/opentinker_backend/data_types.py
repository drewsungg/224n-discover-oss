"""
Replacement data types for the tinker API.

These plain Python/PyTorch types replace tinker.ModelInput, tinker.Datum,
tinker.TensorData, tinker.SamplingParams, etc.
"""

from dataclasses import dataclass, field

import torch


@dataclass
class EncodedTextChunk:
    """Replaces tinker.types.EncodedTextChunk — a chunk of encoded tokens."""

    tokens: list[int]

    @property
    def length(self) -> int:
        return len(self.tokens)


# In TTT-Discover, all ModelInputChunks are text-only (no images).
ModelInputChunk = EncodedTextChunk


@dataclass
class TokenSequence:
    """Replaces tinker.ModelInput — a sequence of token IDs.

    Unlike tinker.ModelInput which uses a list of chunks, this is a flat
    token list since TTT-Discover only uses text (no images).
    """

    tokens: list[int]

    @staticmethod
    def empty() -> "TokenSequence":
        return TokenSequence(tokens=[])

    def append_int(self, token: int) -> "TokenSequence":
        return TokenSequence(tokens=self.tokens + [token])

    @property
    def length(self) -> int:
        return len(self.tokens)

    @property
    def chunks(self) -> list[EncodedTextChunk]:
        """Compatibility property for code that accesses .chunks."""
        return [EncodedTextChunk(tokens=self.tokens)]

    @staticmethod
    def from_chunks(chunks: list[EncodedTextChunk]) -> "TokenSequence":
        """Build a TokenSequence from a list of EncodedTextChunks."""
        tokens: list[int] = []
        for chunk in chunks:
            tokens.extend(chunk.tokens)
        return TokenSequence(tokens=tokens)


@dataclass
class TrainingDatum:
    """Replaces tinker.Datum.

    Holds a single training example with model input and associated
    loss function inputs (target tokens, logprobs, advantages, mask).
    """

    model_input: TokenSequence
    loss_fn_inputs: dict[str, torch.Tensor]


@dataclass
class SamplingParams:
    """Replaces tinker.SamplingParams."""

    stop: list[str] | list[int] = field(default_factory=list)
    max_tokens: int = 1024
    temperature: float = 1.0


@dataclass
class SampleSequence:
    """A single generated sequence with per-token logprobs."""

    tokens: list[int]
    logprobs: list[float]


@dataclass
class SampleResult:
    """Result of a sampling call. Replaces tinker sample_async return."""

    sequences: list[SampleSequence]
