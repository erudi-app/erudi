"""The frozen e5 tokenizer that counts tokens for KB chunking and retrieval.

``src/ingestion/e5_tokenizer.json.gz`` is the ``intfloat/multilingual-e5-small``
tokenizer as ``transformers`` builds it (``backend_tokenizer.to_str()``), loaded
with the pure ``tokenizers`` package. These tests hold it to the tokenizer it
replaces while ``transformers`` is still installed: same ids on varied text,
and the same chunks out of the 3-pass chunker.
"""

import sys
from pathlib import Path

import pytest

from src.core import config
from src.ingestion import chunking
from src.ingestion.embedding_model import EMBEDDING_MODEL_ID

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]

PARITY_TEXTS = [
    "",
    " ",
    "OK",
    "42",
    "€",
    "?",
    "café",
    "Le café des développeurs était fermé, l'équipe a donc travaillé à la maison.",
    "Quel est le préavis de résiliation prévu par le contrat Nimbus ?",
    "The quarterly security audit found no critical vulnerabilities.",
    "日本語のテキストと中文文本混合在一起。数据库迁移在启动时自动执行。",
    "Привет, как дела? Ελληνικά κείμενα. مرحبا بالعالم",
    "emoji 🚀🔥 and  double  spaces\tand tabs\nand\nnewlines\n\n\nand more",
    "def count_tokens(text: str) -> int:\n    return len(tok.encode(text).ids)  # noqa\n",
    "const x = await fetch(`${API_BASE_URL}/erudi/llms`, { method: 'GET' });",
    "| Article | Prix |\n| --- | --- |\n| Produit 1 | 100 € |\n| Produit 2 | 200 € |",
    "# Guide > ## Installation\n\nÉtapes détaillées pour installer le produit.",
    "A literal <mask> token, a <s> start and a </s> end, plus <pad> and <unk>.",
    "FAC-2024-0917 a été réglée le 15/03/2024 à 14h32 (UTC+01:00).",
    " non-breaking space and zero​width and ligature ﬁ and fullwidth ＡＢＣ",
    "Ünïcödé ñ ç ß ø å æ œ — “quotes” ‘single’ … ellipsis",
]


def _long_text() -> str:
    return (REPO_ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")


def _cached_snapshot() -> "Path | None":
    """A local e5 snapshot holding the tokenizer, from either cache that
    keeps one: the app's model cache (the macOS embedding child's snapshot)
    or the GGUF build script's own cache (``backend/data/e5-build-cache``,
    where a Windows/Linux dev tree has it)."""
    folder = "models--" + EMBEDDING_MODEL_ID.replace("/", "--")
    for cache in (config.CACHE_DIR, config.ROOT_DIR / "data" / "e5-build-cache"):
        for tokenizer in sorted((Path(cache) / folder / "snapshots").glob("*/tokenizer.json")):
            return tokenizer.parent
    return None


@pytest.fixture(scope="module")
def reference_tokenizer():
    """The tokenizer chunking used before: transformers over the e5 snapshot.

    Read from a local cache only, never downloaded: without the snapshot (an
    offline machine, a fresh checkout) the comparisons skip.
    """
    snapshot = _cached_snapshot()
    if snapshot is None:
        pytest.skip("no local e5 snapshot (models_cache or data/e5-build-cache)")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)


@pytest.fixture(scope="module")
def frozen_tokenizer():
    return chunking._load_frozen_tokenizer()


class TestFrozenFile:
    def test_frozen_file_ships_inside_the_backend_tree(self):
        assert chunking.E5_TOKENIZER_FILE.is_file()
        assert chunking.E5_TOKENIZER_FILE == (
            config.ROOT_DIR / "src" / "ingestion" / "e5_tokenizer.json.gz"
        )

    def test_frozen_tokenizer_has_no_truncation_nor_padding(self, frozen_tokenizer):
        # A count must be the whole text's, never capped at e5's 512 window.
        assert frozen_tokenizer.truncation is None
        assert frozen_tokenizer.padding is None
        assert chunking.count_tokens("mot " * 2000) == 2000

    def test_loading_never_touches_transformers_nor_the_hub(self, monkeypatch):
        # The privacy promise and the coming removal of transformers: the
        # count works offline, with no Hugging Face client involved.
        monkeypatch.setitem(sys.modules, "transformers", None)
        monkeypatch.setattr(chunking, "_tokenizer", None)
        assert chunking.count_tokens("Le café des développeurs était fermé.") > 0


class TestParityWithTransformers:
    @pytest.mark.parametrize("text", PARITY_TEXTS + ["LONG"], ids=lambda t: repr(t[:20]))
    def test_same_ids_and_same_tokenize_length(self, text, reference_tokenizer, frozen_tokenizer):
        if text == "LONG":
            text = _long_text()
        expected = reference_tokenizer.encode(text, add_special_tokens=False)
        assert frozen_tokenizer.encode(text, add_special_tokens=False).ids == expected
        # LangChain's from_huggingface_tokenizer measured with tokenize().
        assert chunking.count_tokens(text) == len(reference_tokenizer.tokenize(text))


def _corpus() -> dict[str, str]:
    files = [
        REPO_ROOT / "backend" / "tests" / "fixtures" / "kb" / "nimbus_contract.txt",
        REPO_ROOT / "docs" / "architecture.md",
        REPO_ROOT / "docs" / "guides" / "knowledge_base.md",
        REPO_ROOT / "docs" / "usage.md",
        REPO_ROOT / "docs" / "privacy.md",
        REPO_ROOT / "BUILD.md",
        REPO_ROOT / "QA-SCENARIOS.md",
        REPO_ROOT / "backend" / "src" / "ingestion" / "chunking.py",
    ]
    corpus = {path.name: path.read_text(encoding="utf-8") for path in files}
    rows = "\n".join(
        f"| Produit numéro {i} avec un libellé assez long | {i * 100} € | 产品 {i} |"
        for i in range(80)
    )
    corpus["table.md"] = f"# Catalogue\n\n| Article | Prix | 名称 |\n| --- | --- | --- |\n{rows}"
    corpus["zh.md"] = "# 文档\n\n" + "\n\n".join(
        f"第{i}段：知识库把文档切分成大约一百八十个词元的片段，并保留百分之十五的重叠。"
        for i in range(40)
    )
    return corpus


class TestChunkBoundariesUnchanged:
    """The chunker gives the chunks the transformers-backed one gave."""

    @pytest.mark.parametrize("budget", [(180, 27), (64, 8)], ids=["default", "small"])
    def test_same_chunks_as_the_transformers_splitter(
        self, budget, reference_tokenizer, monkeypatch
    ):
        target, overlap = budget
        corpus = _corpus()
        current = {
            name: chunking.chunk_markdown(text, target_tokens=target, overlap_tokens=overlap)
            for name, text in corpus.items()
        }

        from langchain_text_splitters import RecursiveCharacterTextSplitter

        def reference_splitter(target_tokens, overlap_tokens):
            return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                reference_tokenizer,
                chunk_size=target_tokens,
                chunk_overlap=overlap_tokens,
                separators=["\n\n", "\n", ". ", " ", ""],
            )

        monkeypatch.setattr(chunking, "_make_sub_splitter", reference_splitter)
        monkeypatch.setattr(
            chunking,
            "count_tokens",
            lambda text: len(reference_tokenizer.encode(text, add_special_tokens=False)),
        )
        for name, text in corpus.items():
            reference = chunking.chunk_markdown(text, target_tokens=target, overlap_tokens=overlap)
            assert len(reference) > 0, name
            assert current[name] == reference, name
