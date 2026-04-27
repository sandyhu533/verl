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
"""
The base tokenizer class, required for any hybrid engine based rollout or inference with vLLM.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch

__all__ = ["HybridEngineBaseTokenizer"]


class HybridEngineBaseTokenizer(ABC):
    """Abstract tokenizer surface shared by veRL training code and rollout engines.

    What:
      - Duck-typed contract that any tokenizer handed to a HybridEngine
        rollout (vLLM, SGLang, TRT-LLM) must satisfy. Signatures and
        property names are intentionally aligned with HuggingFace's
        `PreTrainedTokenizer(Fast)` because vLLM and friends introspect
        those exact names (`pad_token_id`, `eos_token_id`,
        `all_special_ids`, `encode`, `decode`, `convert_ids_to_tokens`,
        `get_added_vocab`, `convert_tokens_to_string`).
      - Lets veRL wrap non-HF tokenizers (custom BPE, processor-only
        multimodal paths) and still plug them into HF-shaped call sites
        without monkey-patching.

    Lifecycle:
      - Pure abstract -- never instantiated. Concrete subclasses live in
        model-specific tokenizer shims; most call sites fall back to
        `transformers.AutoTokenizer` directly, and this ABC is the
        escape hatch when they cannot.
      - `is_fast=False` by default: subclasses that back a Rust
        `tokenizers` instance override it so downstream code can pick
        the faster batch paths.

    Called by:
      - Rollout boundary validation: `AsyncRolloutRequest` performs a
        tokenization sanity check (see `TokenizationSanityCheckModeEnum`
        in `verl.workers.rollout.schemas`) that re-encodes the assembled
        prompt via this interface and compares token ids with the
        ids actually fed to the engine.
      - `apply_chat_template` consumers in the rollout chat path rely on
        this being HF-shaped so the same template works across engines.

    Branches:
      - `is_fast` flips batch-encode vs single-encode code paths on
        callers.
      - Properties returning `Optional[int]` (pad/eos) force callers to
        handle models that genuinely have no pad token.

    Why:
      - veRL's hybrid engine must agree with the training-side tokenizer
        byte-for-byte; a mismatch shifts labels by one token and
        silently corrupts PPO/GRPO advantages. Pinning the interface
        here -- rather than accepting `Any` -- makes that contract
        explicit and gives one place to add veRL-specific defaults
        (e.g. chat-template normalization) across rollout backends.
      - Keeping names identical to HF (instead of a cleaner veRL-native
        API) is a deliberate compatibility choice: vLLM's `LLMEngine`
        and TRT-LLM's OpenAI-compatible server both call these methods
        by name, so any rename would fork engine code.
    """

    @property
    @abstractmethod
    def vocab_size(self):
        """
        `int`: Size of the base vocabulary (without the added tokens).
        """
        pass

    @property
    @abstractmethod
    def pad_token_id(self):
        """
        `Optional[int]`: Id of the padding token in the vocabulary. Returns `None` if the token has not been set.
        """
        pass

    @property
    @abstractmethod
    def eos_token_id(self):
        """
        `Optional[int]`: Id of the end of sentence token in the vocabulary. Returns `None` if the token has not been
        set.
        """
        pass

    @property
    @abstractmethod
    def all_special_ids(self) -> list[int]:
        """
        `List[int]`: List the ids of the special tokens(`'<unk>'`, `'<cls>'`, etc.) mapped to class attributes.
        """
        pass

    @property
    @abstractmethod
    def all_special_tokens(self) -> list[str]:
        """
        `List[str]`: A list of the unique special tokens (`'<unk>'`, `'<cls>'`, ..., etc.).

        Convert tokens of `tokenizers.AddedToken` type to string.
        """
        pass

    @abstractmethod
    def encode(self, text):
        """
        Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.

        Args:
            text (`str`, `List[str]` or `List[int]`):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
                `tokenize` method) or a list of integers.

            text_pair (`str`, `List[str]` or `List[int]`, *optional*):
                Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers.
        """
        pass

    @abstractmethod
    def decode(
        self,
        token_ids: int | list[int] | np.ndarray | torch.Tensor,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> str:
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

        Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces`.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `str`: The decoded sentence.
        """
        pass

    @abstractmethod
    def convert_ids_to_tokens(self, ids: int | list[int], skip_special_tokens: bool = False) -> str | list[str]:
        """
        Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
        added tokens.

        Args:
            ids (`int` or `List[int]`):
                The token id (or token ids) to convert to tokens.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.

        Returns:
            `str` or `List[str]`: The decoded token(s).
        """
        pass

    @abstractmethod
    def get_added_vocab(self) -> dict[str, int]:
        """
        Returns the added tokens in the vocabulary as a dictionary of token to index. Results might be different from
        the fast call because for now we always add the tokens even if they are already in the vocabulary. This is
        something we should change.

        Returns:
            `Dict[str, int]`: The added tokens.
        """
        pass

    @abstractmethod
    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """
        Converts a sequence of tokens in a single string. The most simple way to do it is `" ".join(tokens)` but we
        often want to remove sub-word tokenization artifacts at the same time.

        Args:
            tokens (`List[str]`): The token to join in a string.

        Returns:
            `str`: The joined tokens.
        """
        pass

    @property
    def is_fast(self):
        return False
