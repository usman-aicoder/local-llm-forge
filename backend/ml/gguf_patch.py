"""
Wrapper for llama.cpp convert_hf_to_gguf.py that patches the Qwen2 tokenizer bug.

The bug: newer transformers passes extra_special_tokens as a list in Qwen2's
__init__, but PreTrainedTokenizerBase._set_model_specific_special_tokens expects
a dict. This patch converts the list to {} before the error is triggered.

Usage: python gguf_patch.py <convert_script> [convert_args...]
"""
from __future__ import annotations

import sys


def _patch_transformers() -> None:
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase

        orig = PreTrainedTokenizerBase._set_model_specific_special_tokens

        def patched(self, special_tokens=None):  # type: ignore[override]
            if isinstance(special_tokens, list):
                special_tokens = {}
            return orig(self, special_tokens=special_tokens)

        PreTrainedTokenizerBase._set_model_specific_special_tokens = patched  # type: ignore[method-assign]
    except Exception:
        pass  # best-effort — if it fails the original error will surface


_patch_transformers()

# Shift argv so the convert script sees itself as argv[0]
convert_script = sys.argv[1]
sys.argv = sys.argv[1:]

import runpy  # noqa: E402
runpy.run_path(convert_script, run_name="__main__")
