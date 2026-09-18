# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FastIncrementalDetokenizer primes DecodeStream from a tail of the prompt.

DecodeStream decodes its whole id list on the first step to establish a
baseline, so priming with the entire prompt puts an O(prompt length) decode on
the first generated token -- the one TTFT measures. The primed ids only fix the
boundary for that first delta: output_text starts empty, stop strings match
generated text only, and echo carries prompt_text separately. So a tail is
enough, and SlowIncrementalDetokenizer already assumes it.

These pin both halves of that: the emitted text must not change, and the tail
must actually be what gets primed.
"""

import pytest
import tokenizers
from transformers import AutoTokenizer, TokenizersBackend

import vllm.v1.engine.detokenizer as detok_mod
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.detokenizer import FastIncrementalDetokenizer

TOKENIZERS = ["openai-community/gpt2", "facebook/opt-125m", "EleutherAI/pythia-70m"]

# CJK and emoji are the cases that could break: byte-level BPE splits a single
# character across tokens, so a boundary the window misses would corrupt output.
TEXTS = [
    "The quick brown fox jumps over the lazy dog. " * 30,
    "深度学习模型的推理延迟主要由访存带宽决定，而不是算力。" * 30,
    "🚀🔥😀 mixed emoji and 中文 and English 混排 " * 30,
    "def f(x):\n    return x ** 2 + 1  # comment\n" * 30,
]


def _request(prompt_token_ids, skip_special_tokens=True):
    return EngineCoreRequest(
        request_id="",
        prompt_token_ids=prompt_token_ids,
        mm_features=None,
        sampling_params=SamplingParams(skip_special_tokens=skip_special_tokens),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def _decode(tokenizer, prompt_ids, gen_ids):
    d = FastIncrementalDetokenizer(tokenizer, _request(prompt_ids))
    d.update(list(gen_ids), False)
    return d.get_next_output_text(finished=True, delta=False)


@pytest.fixture
def tokenizer(tokenizer_name):
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    if not isinstance(tok, TokenizersBackend):
        pytest.skip(f"{tokenizer_name} has no fast backend")
    return tok


@pytest.mark.parametrize("tokenizer_name", TOKENIZERS)
@pytest.mark.parametrize("text", TEXTS)
def test_tail_priming_emits_the_same_text(tokenizer, text, monkeypatch):
    """A tail-primed stream must emit byte-identical output to a fully primed one."""
    ids = tokenizer(text, add_special_tokens=False).input_ids
    assert len(ids) > 64, "prompt must be longer than the priming window"
    prompt_ids, gen_ids = ids[:-8], ids[-8:]

    monkeypatch.setattr(detok_mod, "_PRIME_TAIL_TOKENS", 0)
    full = _decode(tokenizer, prompt_ids, gen_ids)

    for tail in (4, 8, 32, 128):
        monkeypatch.setattr(detok_mod, "_PRIME_TAIL_TOKENS", tail)
        assert _decode(tokenizer, prompt_ids, gen_ids) == full, f"tail={tail}"


@pytest.mark.parametrize("tokenizer_name", ["openai-community/gpt2"])
def test_output_never_contains_the_prompt(tokenizer):
    """Priming seeds the boundary; it must not put prompt text in the output."""
    ids = tokenizer(TEXTS[0], add_special_tokens=False).input_ids
    prompt_ids, gen_ids = ids[:-4], ids[-4:]
    out = _decode(tokenizer, prompt_ids, gen_ids)
    assert out == tokenizer.decode(gen_ids)


@pytest.mark.parametrize("tokenizer_name", ["openai-community/gpt2"])
def test_only_the_tail_is_primed(tokenizer, monkeypatch):
    """The point of the change: the long prompt must not reach DecodeStream."""
    seen = {}
    real = tokenizers.decoders.DecodeStream

    def spy(ids, skip_special_tokens):
        seen["ids"] = list(ids)
        return real(ids=ids, skip_special_tokens=skip_special_tokens)

    monkeypatch.setattr(tokenizers.decoders, "DecodeStream", spy)
    prompt_ids = tokenizer(TEXTS[0], add_special_tokens=False).input_ids

    monkeypatch.setattr(detok_mod, "_PRIME_TAIL_TOKENS", 32)
    FastIncrementalDetokenizer(tokenizer, _request(prompt_ids))
    assert seen["ids"] == prompt_ids[-32:]

    monkeypatch.setattr(detok_mod, "_PRIME_TAIL_TOKENS", 0)
    FastIncrementalDetokenizer(tokenizer, _request(prompt_ids))
    assert seen["ids"] == prompt_ids, "0 must restore priming with the whole prompt"


@pytest.mark.parametrize("tokenizer_name", ["openai-community/gpt2"])
def test_prompt_shorter_than_the_window_is_untouched(tokenizer, monkeypatch):
    seen = {}
    real = tokenizers.decoders.DecodeStream

    def spy(ids, skip_special_tokens):
        seen["ids"] = list(ids)
        return real(ids=ids, skip_special_tokens=skip_special_tokens)

    monkeypatch.setattr(tokenizers.decoders, "DecodeStream", spy)
    monkeypatch.setattr(detok_mod, "_PRIME_TAIL_TOKENS", 32)

    short = tokenizer("hello there", add_special_tokens=False).input_ids
    assert len(short) < 32
    FastIncrementalDetokenizer(tokenizer, _request(short))
    assert seen["ids"] == short

    FastIncrementalDetokenizer(tokenizer, _request([]))
    assert seen["ids"] == []
