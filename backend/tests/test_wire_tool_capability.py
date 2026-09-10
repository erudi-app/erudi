"""Engine hooks for the verified tool-call WIRE capability (#298).

``supports_tools`` says the chat template DECLARES tools; the wire capability
says the engine's server actually turns the model's tool-call output into a
structured call the agent executes. That is a per-model wire property (#273
hardware matrix): on mlx-vlm the server infers a tool parser from chat-template
markers and a template matching NO parser streams the call as raw text (#295),
while llama.cpp with ``--jinja`` hands an unmatched template to its
differential autoparser, so any usable template gets structured tool handling.

- MLX: ``compute_wire_tools`` runs mlx-vlm's OWN ``_infer_tool_parser`` on the
  model's chat template -> True iff a parser is returned. Same code the server
  executes, exact by construction for the pinned mlx-vlm.
- llama.cpp: template loads -> True (specialized handler or autoparser); no
  template -> False; unreadable artifact -> None.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from src.engines.base_engine import BaseEngine
from src.engines.cpu_engine import CPU_Engine
from src.engines.mlx_engine import MLX_Engine

pytestmark = pytest.mark.unit


# A Qwen3-style template: carries BOTH json_tools markers mlx-vlm 0.6.13
# requires ("<tool_call>" and "tool_call.name"), so a parser is inferred.
QWEN3_STYLE_TEMPLATE = (
    "{%- if tools %}{%- for tool_call in message.tool_calls %}"
    "<tool_call>\n"
    '{"name": "{{ tool_call.name }}", "arguments": {{ tool_call.arguments }}}'
    "\n</tool_call>{%- endfor %}{%- endif %}"
)

# A Qwen3-4B-2507-style template: emits tool calls with markers matching NO
# entry of the mlx-vlm parser table (#295) -> no parser -> raw text on the wire.
NO_PARSER_TEMPLATE = (
    "{%- if tools %}{%- for fc in message.tool_calls %}"
    "<function_call>{{ fc.name }}({{ fc.arguments }})</function_call>"
    "{%- endfor %}{%- endif %}"
)


def _fake_mlx_vlm(monkeypatch, infer):
    """Install a fake ``mlx_vlm.tool_parsers`` exposing ``_infer_tool_parser``."""
    pkg = types.ModuleType("mlx_vlm")
    mod = types.ModuleType("mlx_vlm.tool_parsers")
    mod._infer_tool_parser = infer
    pkg.tool_parsers = mod
    monkeypatch.setitem(sys.modules, "mlx_vlm", pkg)
    monkeypatch.setitem(sys.modules, "mlx_vlm.tool_parsers", mod)


def _tokenizer_with(template):
    return SimpleNamespace(chat_template=template)


# ---------------- BaseEngine default ----------------


def test_base_engine_default_is_unknown():
    assert BaseEngine.compute_wire_tools("/m/path") is None


# ---------------- MLX hook (fake mlx_vlm: contract) ----------------


class TestMlxComputeWireTools:
    def test_true_when_a_parser_is_inferred(self, monkeypatch):
        _fake_mlx_vlm(monkeypatch, lambda t: "json_tools" if "tool_call.name" in t else None)
        monkeypatch.setattr(
            MLX_Engine,
            "_load_capability_tokenizer",
            classmethod(lambda cls, p: _tokenizer_with(QWEN3_STYLE_TEMPLATE)),
        )
        assert MLX_Engine.compute_wire_tools("/m/qwen3") is True

    def test_false_when_no_parser_matches(self, monkeypatch):
        _fake_mlx_vlm(monkeypatch, lambda t: None)
        monkeypatch.setattr(
            MLX_Engine,
            "_load_capability_tokenizer",
            classmethod(lambda cls, p: _tokenizer_with(NO_PARSER_TEMPLATE)),
        )
        assert MLX_Engine.compute_wire_tools("/m/qwen3-2507") is False

    def test_false_when_template_missing(self, monkeypatch):
        # No chat template -> _infer_tool_parser(None) -> no parser -> False.
        _fake_mlx_vlm(
            monkeypatch,
            lambda t: "json_tools" if isinstance(t, str) and "tool_call.name" in t else None,
        )
        monkeypatch.setattr(
            MLX_Engine,
            "_load_capability_tokenizer",
            classmethod(lambda cls, p: _tokenizer_with(None)),
        )
        assert MLX_Engine.compute_wire_tools("/m/no-template") is False

    def test_none_when_mlx_vlm_import_fails(self, monkeypatch):
        broken = types.ModuleType("mlx_vlm")  # no tool_parsers attribute
        monkeypatch.setitem(sys.modules, "mlx_vlm", broken)
        monkeypatch.delitem(sys.modules, "mlx_vlm.tool_parsers", raising=False)
        monkeypatch.setattr(
            MLX_Engine,
            "_load_capability_tokenizer",
            classmethod(lambda cls, p: _tokenizer_with(QWEN3_STYLE_TEMPLATE)),
        )
        assert MLX_Engine.compute_wire_tools("/m/path") is None

    def test_none_when_tokenizer_load_fails(self, monkeypatch):
        _fake_mlx_vlm(monkeypatch, lambda t: "json_tools")

        def _boom(cls, p):
            raise RuntimeError("unreadable artifact")

        monkeypatch.setattr(MLX_Engine, "_load_capability_tokenizer", classmethod(_boom))
        assert MLX_Engine.compute_wire_tools("/m/path") is None

    def test_none_when_inference_itself_raises(self, monkeypatch):
        def _raise(t):
            raise TypeError("unexpected template object")

        _fake_mlx_vlm(monkeypatch, _raise)
        monkeypatch.setattr(
            MLX_Engine,
            "_load_capability_tokenizer",
            classmethod(lambda cls, p: _tokenizer_with(QWEN3_STYLE_TEMPLATE)),
        )
        assert MLX_Engine.compute_wire_tools("/m/path") is None


# ---------------- MLX hook against the REAL mlx-vlm (Mac only) ----------------


class TestMlxComputeWireToolsRealMlxVlm:
    """Run the hook against the real pinned mlx-vlm when it is installed.

    Proves the hook executes the exact inference the server executes: the
    Qwen3-style template matches json_tools, the 2507-style one matches nothing.
    Skipped cleanly on hosts without mlx-vlm (Linux CI).
    """

    def test_qwen3_style_template_gets_a_parser(self, monkeypatch):
        pytest.importorskip("mlx_vlm")
        monkeypatch.setattr(
            MLX_Engine,
            "_load_capability_tokenizer",
            classmethod(lambda cls, p: _tokenizer_with(QWEN3_STYLE_TEMPLATE)),
        )
        assert MLX_Engine.compute_wire_tools("/m/qwen3") is True

    def test_2507_style_template_gets_no_parser(self, monkeypatch):
        pytest.importorskip("mlx_vlm")
        monkeypatch.setattr(
            MLX_Engine,
            "_load_capability_tokenizer",
            classmethod(lambda cls, p: _tokenizer_with(NO_PARSER_TEMPLATE)),
        )
        assert MLX_Engine.compute_wire_tools("/m/qwen3-2507") is False


# ---------------- llama.cpp hook ----------------


class TestLlamaComputeWireTools:
    def test_true_with_a_chat_template(self, monkeypatch):
        # Any usable template -> True: with --jinja an unmatched template is
        # still handed to the autoparser, which generates a parser for it.
        monkeypatch.setattr(
            CPU_Engine,
            "_load_capability_tokenizer",
            classmethod(lambda cls, p: _tokenizer_with(NO_PARSER_TEMPLATE)),
        )
        assert CPU_Engine.compute_wire_tools("/m/model.gguf") is True

    def test_false_without_a_chat_template(self, monkeypatch):
        monkeypatch.setattr(
            CPU_Engine,
            "_load_capability_tokenizer",
            classmethod(lambda cls, p: _tokenizer_with(None)),
        )
        assert CPU_Engine.compute_wire_tools("/m/model.gguf") is False

    def test_none_when_tokenizer_load_fails(self, monkeypatch):
        def _boom(cls, p):
            raise RuntimeError("gguf unreadable")

        monkeypatch.setattr(CPU_Engine, "_load_capability_tokenizer", classmethod(_boom))
        assert CPU_Engine.compute_wire_tools("/m/model.gguf") is None


class TestLlamaNativeFormatTable:
    """The mirrored chat.cpp marker table is for LOGS only, never the verdict."""

    def test_table_mirrors_the_dispatch_order(self):
        from src.engines.base_llama_cpp_engine import LLAMA_NATIVE_TOOL_FORMATS

        # Every entry is (name, required, forbidden, any_of); chat.cpp tests
        # them in this order, and a required group is never empty.
        for name, required, forbidden, any_of in LLAMA_NATIVE_TOOL_FORMATS:
            assert isinstance(name, str) and name
            assert required, f"{name} would match every template"
            for group in (required, forbidden, any_of):
                assert all(isinstance(m, str) and m for m in group)

    def test_qwen3_coder_needs_all_three_markers(self):
        from src.engines.base_llama_cpp_engine import native_tool_format_for_template

        tpl = "... <tool_call> <function=foo> <parameter=bar> ..."
        assert native_tool_format_for_template(tpl) == "qwen3_coder"

    def test_a_lone_tool_call_marker_is_not_a_specialized_format(self):
        from src.engines.base_llama_cpp_engine import native_tool_format_for_template

        # <tool_call> alone matched a native handler before the b10883 dispatch;
        # it now needs the <function=>/<parameter=> pair to be Qwen3-Coder.
        assert native_tool_format_for_template("... <tool_call> ...") == "autoparser"

    def test_forbidden_marker_rules_a_format_out(self):
        from src.engines.base_llama_cpp_engine import native_tool_format_for_template

        # Mistral Small 3.2 carries [CALL_ID]; Ministral is the one that does not.
        ministral = "[SYSTEM_PROMPT] [TOOL_CALLS] [ARGS]"
        assert native_tool_format_for_template(ministral) == "ministral_3"
        assert native_tool_format_for_template(ministral + " [CALL_ID]") == "autoparser"

    def test_any_of_group_accepts_either_marker(self):
        from src.engines.base_llama_cpp_engine import native_tool_format_for_template

        base = "dsml_token DSML "
        assert native_tool_format_for_template(base + "function_calls") == "deepseek_v3_2"
        assert native_tool_format_for_template(base + "tool_calls") == "deepseek_v3_2"
        # Neither name present -> the any-of group is not satisfied.
        assert native_tool_format_for_template(base) == "autoparser"

    def test_unmatched_template_falls_back_to_the_autoparser(self):
        from src.engines.base_llama_cpp_engine import native_tool_format_for_template

        assert native_tool_format_for_template(NO_PARSER_TEMPLATE) == "autoparser"

    def test_generic_never_flips_the_verdict(self, monkeypatch):
        # A template no native handler matches is still wire-capable on llama.
        monkeypatch.setattr(
            CPU_Engine,
            "_load_capability_tokenizer",
            classmethod(lambda cls, p: _tokenizer_with("plain {{ messages }}")),
        )
        assert CPU_Engine.compute_wire_tools("/m/model.gguf") is True
