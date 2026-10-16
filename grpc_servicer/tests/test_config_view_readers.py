"""The readers the engines' configuration views share: a count is a positive
integer and nothing else; an HF-config attribute reads leniently, with the
default when the config or the attribute is absent."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from smg_grpc_servicer.config_view import hf_int, hf_list, hf_str, positive_int


@pytest.mark.parametrize("value", [1, 8, 2**40])
def test_a_positive_integer_is_a_count(value):
    assert positive_int(value) == value


@pytest.mark.parametrize("value", [0, -1, True, False, None, "4", 4.0, [4]])
def test_anything_else_is_not_a_count(value):
    assert positive_int(value) is None


def test_hf_attributes_read_leniently():
    hf_config = SimpleNamespace(
        model_type="llama", architectures=["LlamaForCausalLM"], pad_token_id=7
    )
    assert hf_str(hf_config, "model_type") == "llama"
    assert hf_list(hf_config, "architectures") == ["LlamaForCausalLM"]
    assert hf_list(hf_config, "architectures") is not hf_config.architectures, "a copy"
    assert hf_int(hf_config, "pad_token_id") == 7
    # absent attributes
    assert hf_str(hf_config, "revision") == ""
    assert hf_list(hf_config, "eos_token_ids") == []
    assert hf_int(hf_config, "bos_token_id") == 0
    # falsy values read as the default too, as the sections always did
    assert hf_str(SimpleNamespace(model_type=None), "model_type") == ""
    assert hf_int(SimpleNamespace(pad_token_id=None), "pad_token_id") == 0


@pytest.mark.parametrize("reader, default", [(hf_str, ""), (hf_list, []), (hf_int, 0)])
def test_no_config_reads_as_the_default(reader, default):
    assert reader(None, "anything") == default
