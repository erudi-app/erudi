"""The CPU engine's spawn window against a REAL llama-server.

A tiny random-weight llama GGUF is written on the fly; the CPU engine sizes
its window against a pinned RAM total small enough for memory to bind, spawns
the built llama-server, and reads back what the child allocated: ``/props``
must report exactly the window passed as ``-c``, and the post-load check of
the child's anonymous memory must stay quiet. Skipped where llama.cpp is not
built (the CI legs do not build it) or the ``gguf`` package is absent.
"""

from __future__ import annotations

import logging
import os
import socket

import pytest

from src.engines import gguf_memory, memory_pressure
from src.engines.cpu_engine import CPU_Engine

pytestmark = pytest.mark.integration

GIB = 1024**3
TRAINED = 131_072


def _binary():
    exe = "llama-server.exe" if os.name == "nt" else "llama-server"
    path = CPU_Engine._default_install_dir() / exe
    if not path.exists():
        pytest.skip(f"llama-server not built at {path}")
    return path


def _free_port(start=28100):
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    pytest.skip("no free port in 28100-28199")


def _tiny_gguf(path):
    gguf = pytest.importorskip("gguf")
    import numpy as np

    vocab, embd, heads, kv_heads, head_dim, ff = 259, 64, 4, 2, 128, 128
    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_block_count(1)
    writer.add_context_length(TRAINED)
    writer.add_embedding_length(embd)
    writer.add_feed_forward_length(ff)
    writer.add_head_count(heads)
    writer.add_head_count_kv(kv_heads)
    writer.add_key_length(head_dim)
    writer.add_value_length(head_dim)
    writer.add_rope_dimension_count(head_dim)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_tokenizer_model("llama")
    # Byte tokens, "A" first: with a zero output matrix every logit ties and
    # greedy decoding picks token 0, so the readiness ping reads a valid "A".
    order = [0x41] + [i for i in range(256) if i != 0x41]
    writer.add_token_list([f"<0x{i:02X}>".encode() for i in order] + [b"<unk>", b"<s>", b"</s>"])
    writer.add_token_scores([0.0] * vocab)
    writer.add_token_types([6] * 256 + [2, 3, 3])
    writer.add_unk_token_id(256)
    writer.add_bos_token_id(257)
    writer.add_eos_token_id(258)
    writer.add_chat_template("{% for m in messages %}{{ m['content'] }}{% endfor %}")
    rng = np.random.default_rng(0)

    def tensor(name, *shape):
        writer.add_tensor(name, (rng.standard_normal(shape) * 0.02).astype(np.float32))

    tensor("token_embd.weight", vocab, embd)
    writer.add_tensor("blk.0.attn_norm.weight", np.ones(embd, dtype=np.float32))
    tensor("blk.0.attn_q.weight", heads * head_dim, embd)
    tensor("blk.0.attn_k.weight", kv_heads * head_dim, embd)
    tensor("blk.0.attn_v.weight", kv_heads * head_dim, embd)
    tensor("blk.0.attn_output.weight", embd, heads * head_dim)
    writer.add_tensor("blk.0.ffn_norm.weight", np.ones(embd, dtype=np.float32))
    tensor("blk.0.ffn_gate.weight", ff, embd)
    tensor("blk.0.ffn_up.weight", ff, embd)
    tensor("blk.0.ffn_down.weight", embd, ff)
    writer.add_tensor("output_norm.weight", np.ones(embd, dtype=np.float32))
    writer.add_tensor("output.weight", np.zeros((vocab, embd), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def test_a_real_child_allocates_the_window_the_budget_computed(tmp_path, monkeypatch, caplog):
    _binary()
    model = _tiny_gguf(tmp_path / "tiny-f32.gguf")
    # Half a GiB of "RAM": the budget, not the trained window, binds.
    monkeypatch.setattr(memory_pressure, "effective_total_ram_bytes", lambda: GIB // 2)
    monkeypatch.delenv("ERUDI_CTX", raising=False)
    monkeypatch.delenv("ERUDI_FORCE_MEMORY_PRESSURE", raising=False)
    monkeypatch.setattr(CPU_Engine, "_window_memory", {})
    expected = gguf_memory.spawn_window(model, total_ram_bytes=GIB // 2)
    assert expected.memory_bound and expected.window < TRAINED

    handle = None
    try:
        with caplog.at_level(logging.INFO, logger="erudi"):
            handle = CPU_Engine._start_server(
                model_path=model, alias="erudi-live", port=_free_port()
            )
        assert handle["ctx_size"] == expected.window
        assert handle["context_tokens"] == expected.window
        assert handle["memory_bound_tokens"] == expected.window
        messages = [r.getMessage() for r in caplog.records]
        assert not any("window budget predicted" in m for m in messages)
        assert not any("where -c" in m for m in messages)
    finally:
        if handle is not None:
            CPU_Engine._model = handle
            CPU_Engine.cleanup()
