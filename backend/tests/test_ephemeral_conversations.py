"""Ephemeral conversations (#428) and the pasted images a conversation owns.

An ephemeral conversation is created with ``ephemeral: true``, is never listed
or searched, has no generated title, and is deleted -- row, messages,
checkpointer thread and pasted images -- when the user leaves it, when the app
stops, or at the next start after a crash. A delete waits for a turn still
writing to the conversation (``src.domains.conversations.turns``), so nothing
is written after it.

Run against the real throwaway cluster; pk values are never asserted.
"""

import asyncio
import uuid

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import text

import src.agents.runner as agent_runner
import src.domains.conversations.services as conversation_services
from src.core import config
from src.domains.conversations.ephemeral import (
    purge_ephemeral_conversations,
    purge_orphan_threads,
)
from src.domains.conversations.repository import ConversationRepository
from src.domains.conversations.schemas import ConversationQuery
from src.domains.conversations.services import ConversationService
from src.engines.base_engine import BaseEngine
from src.entities.Conversation import Conversation
from src.entities.Message import Message
from tests._helpers import ToolableFakeChatModel

pytestmark = pytest.mark.integration


class _FakeEngine(BaseEngine):
    """Engine stub exposing the generation guard without a real model."""


def _fake_chat_model(*texts):
    msgs = [AIMessage(content=t) for t in texts]
    return lambda llm, **kw: ToolableFakeChatModel(messages=iter(msgs))


def _thread(conversation_id) -> dict:
    return {"configurable": {"thread_id": str(conversation_id), "checkpoint_ns": ""}}


@pytest.fixture
def pasted_dir(tmp_path, monkeypatch):
    """The app's pasted-images folder, under a throwaway data folder."""
    folder = tmp_path / "data" / "pasted-images"
    folder.mkdir(parents=True)
    monkeypatch.setattr(conversation_services, "pasted_images_dir", lambda: folder)
    return folder


@pytest.fixture(scope="session")
def checkpointer_tables(pg_test_cluster):
    """The LangGraph tables, created before any test transaction opens:
    their setup builds indexes CONCURRENTLY, which waits for every open
    transaction -- the test session's included."""
    from src.agents.checkpoint import open_checkpointer

    async def setup():
        async with open_checkpointer(pg_test_cluster.psycopg_url):
            pass

    asyncio.run(setup())
    return pg_test_cluster


@pytest.fixture
def fake_model(monkeypatch):
    monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
    monkeypatch.setattr(agent_runner, "build_chat_model", _fake_chat_model("An answer here."))


def _marker(path) -> str:
    return f"[image_path:{ConversationService._encode_marker_path(str(path))}]"


# =====================================================================
# Creation, listing, search
# =====================================================================


class TestCreationAndVisibility:
    def test_a_conversation_is_not_ephemeral_unless_asked(self, client, mock_llm):
        response = client.post("/erudi/conversations/", json={"llm_id": mock_llm.id})

        assert response.status_code == 201
        assert response.json()["ephemeral"] is False

    def test_an_ephemeral_conversation_says_so(self, client, mock_llm):
        response = client.post(
            "/erudi/conversations/", json={"llm_id": mock_llm.id, "ephemeral": True}
        )

        assert response.status_code == 201
        assert response.json()["ephemeral"] is True
        detail = client.get(f"/erudi/conversations/{response.json()['id']}")
        assert detail.status_code == 200
        assert detail.json()["ephemeral"] is True

    def test_the_list_leaves_ephemeral_conversations_out(self, client, mock_llm):
        kept = client.post("/erudi/conversations/", json={"llm_id": mock_llm.id}).json()
        hidden = client.post(
            "/erudi/conversations/", json={"llm_id": mock_llm.id, "ephemeral": True}
        ).json()

        ids = {c["id"] for c in client.get("/erudi/conversations/").json()}

        assert kept["id"] in ids
        assert hidden["id"] not in ids

    def test_the_search_leaves_ephemeral_conversations_out(self, client, mock_llm, test_db_session):
        word = f"zeppelin{uuid.uuid4().hex[:8]}"
        kept = client.post("/erudi/conversations/", json={"llm_id": mock_llm.id}).json()
        hidden = client.post(
            "/erudi/conversations/", json={"llm_id": mock_llm.id, "ephemeral": True}
        ).json()
        for conversation in (kept, hidden):
            test_db_session.add(
                Message(conversation_id=conversation["id"], sender="user", content=word)
            )
        test_db_session.commit()

        ids = {c["id"] for c in client.get(f"/erudi/conversations/search?q={word}").json()}

        assert ids == {kept["id"]}

    def test_the_search_by_title_leaves_ephemeral_conversations_out(
        self, test_db_session, mock_llm
    ):
        repo = ConversationRepository(test_db_session)
        title = f"Airships {uuid.uuid4().hex[:8]}"
        hidden = repo.create_conversation(llm_id=mock_llm.id, name=title, ephemeral=True)
        test_db_session.commit()

        assert hidden.id not in {c.id for c in repo.search_conversations(title)}

    def test_a_model_counts_only_the_conversations_a_user_can_find(self, test_db_session, mock_llm):
        from src.domains.llms.repository import Llm_Repository

        repo = ConversationRepository(test_db_session)
        repo.create_conversation(llm_id=mock_llm.id)
        repo.create_conversation(llm_id=mock_llm.id, ephemeral=True)
        test_db_session.commit()

        assert Llm_Repository(test_db_session).count_conversations(mock_llm.id) == 1

    def test_the_ephemeral_flag_cannot_be_changed_later(self, client, mock_llm):
        conversation = client.post("/erudi/conversations/", json={"llm_id": mock_llm.id}).json()

        response = client.patch(
            f"/erudi/conversations/{conversation['id']}", json={"ephemeral": True}
        )

        assert response.status_code == 200
        assert response.json()["ephemeral"] is False


# =====================================================================
# Delete: row, messages, thread, pasted images
# =====================================================================


class TestDelete:
    async def test_delete_removes_row_messages_thread_and_pasted_images(
        self, test_db_session, mock_llm, pasted_dir, fake_model
    ):
        checkpointer = InMemorySaver()
        service = ConversationService(test_db_session, checkpointer)
        conversation = service.create_conversation(llm_id=mock_llm.id, ephemeral=True)
        pasted = pasted_dir / "paste-1.png"
        pasted.write_bytes(b"png")
        payload = ConversationQuery(question="What is this?", images=[], image_paths=[])
        async for _ in service.query_and_respond_stream(conversation.id, payload):
            pass
        test_db_session.add(
            Message(conversation_id=conversation.id, sender="user", content=_marker(pasted))
        )
        test_db_session.commit()
        assert await checkpointer.aget_tuple(_thread(conversation.id)) is not None

        await service.delete_conversation(conversation.id)

        assert test_db_session.get(Conversation, conversation.id) is None
        assert (
            test_db_session.query(Message)
            .filter(Message.conversation_id == conversation.id)
            .count()
            == 0
        )
        assert await checkpointer.aget_tuple(_thread(conversation.id)) is None
        assert not pasted.exists()

    async def test_delete_of_a_regular_conversation_removes_its_pasted_images_too(
        self, test_db_session, mock_llm, pasted_dir
    ):
        service = ConversationService(test_db_session, InMemorySaver())
        conversation = service.create_conversation(llm_id=mock_llm.id)
        pasted = pasted_dir / "paste-2.png"
        pasted.write_bytes(b"png")
        test_db_session.add(
            Message(
                conversation_id=conversation.id,
                sender="user",
                content=f"Look {_marker(pasted)}",
            )
        )
        test_db_session.commit()

        await service.delete_conversation(conversation.id)

        assert not pasted.exists()

    async def test_a_picked_file_is_never_touched(
        self, test_db_session, mock_llm, pasted_dir, tmp_path
    ):
        # An image attached from the disk is referenced by its own path; only
        # a file whose real path is inside pasted-images/ is the app's.
        picked = tmp_path / "Pictures" / "holiday.png"
        picked.parent.mkdir()
        picked.write_bytes(b"png")
        # A name that only starts like the folder, and a path that climbs out
        # of it, are not inside it either.
        sibling = tmp_path / "data" / "pasted-images-old" / "x.png"
        sibling.parent.mkdir()
        sibling.write_bytes(b"png")
        climbing = tmp_path / "data" / "outside.png"
        climbing.write_bytes(b"png")
        service = ConversationService(test_db_session, InMemorySaver())
        conversation = service.create_conversation(llm_id=mock_llm.id, ephemeral=True)
        content = " ".join(
            [_marker(picked), _marker(sibling), _marker(pasted_dir / ".." / "outside.png")]
        )
        test_db_session.add(
            Message(conversation_id=conversation.id, sender="user", content=content)
        )
        test_db_session.commit()

        await service.delete_conversation(conversation.id)

        assert picked.exists()
        assert sibling.exists()
        assert climbing.exists()

    async def test_a_symlink_inside_the_folder_to_a_picked_file_is_not_followed(
        self, test_db_session, mock_llm, pasted_dir, tmp_path
    ):
        picked = tmp_path / "holiday.png"
        picked.write_bytes(b"png")
        link = pasted_dir / "paste-link.png"
        link.symlink_to(picked)
        service = ConversationService(test_db_session, InMemorySaver())
        conversation = service.create_conversation(llm_id=mock_llm.id)
        test_db_session.add(
            Message(conversation_id=conversation.id, sender="user", content=_marker(link))
        )
        test_db_session.commit()

        await service.delete_conversation(conversation.id)

        assert picked.exists()

    async def test_a_pasted_image_another_conversation_shows_is_kept(
        self, test_db_session, mock_llm, pasted_dir
    ):
        service = ConversationService(test_db_session, InMemorySaver())
        leaving = service.create_conversation(llm_id=mock_llm.id, ephemeral=True)
        staying = service.create_conversation(llm_id=mock_llm.id)
        shared = pasted_dir / "paste-shared.png"
        shared.write_bytes(b"png")
        for conversation in (leaving, staying):
            test_db_session.add(
                Message(conversation_id=conversation.id, sender="user", content=_marker(shared))
            )
        test_db_session.commit()

        await service.delete_conversation(leaving.id)

        assert shared.exists()
        assert test_db_session.get(Conversation, staying.id) is not None

    async def test_a_pasted_image_already_gone_is_not_a_failure(
        self, test_db_session, mock_llm, pasted_dir
    ):
        service = ConversationService(test_db_session, InMemorySaver())
        conversation = service.create_conversation(llm_id=mock_llm.id, ephemeral=True)
        test_db_session.add(
            Message(
                conversation_id=conversation.id,
                sender="user",
                content=_marker(pasted_dir / "paste-gone.png"),
            )
        )
        test_db_session.commit()

        await service.delete_conversation(conversation.id)

        assert test_db_session.get(Conversation, conversation.id) is None

    async def test_a_path_with_a_bracket_round_trips_through_its_marker(
        self, test_db_session, mock_llm, pasted_dir
    ):
        pasted = pasted_dir / "paste-[1]%5D.png"
        pasted.write_bytes(b"png")
        service = ConversationService(test_db_session, InMemorySaver())
        conversation = service.create_conversation(llm_id=mock_llm.id)
        test_db_session.add(
            Message(conversation_id=conversation.id, sender="user", content=_marker(pasted))
        )
        test_db_session.commit()

        await service.delete_conversation(conversation.id)

        assert not pasted.exists()


# =====================================================================
# Title
# =====================================================================


class TestTitle:
    async def test_an_ephemeral_conversation_gets_no_title(
        self, test_db_session, mock_llm, monkeypatch
    ):
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
        called = []
        monkeypatch.setattr(
            agent_runner,
            "build_chat_model",
            lambda llm, **kw: called.append(llm) or _fake_chat_model("Secret Plans")(llm),
        )
        service = ConversationService(test_db_session, InMemorySaver())
        conversation = service.create_conversation(llm_id=mock_llm.id, ephemeral=True)

        streamed = [
            t async for t in service.generate_title_stream(conversation.id, "My secret plans")
        ]

        assert streamed == []
        assert called == []
        test_db_session.refresh(conversation)
        assert conversation.name == "New Conversation"


# =====================================================================
# Boot sweep and shutdown purge
# =====================================================================


class TestSweeps:
    async def test_purge_deletes_every_ephemeral_conversation_and_keeps_the_others(
        self, test_db_session, mock_llm, pasted_dir
    ):
        checkpointer = InMemorySaver()
        service = ConversationService(test_db_session, checkpointer)
        left_over = service.create_conversation(llm_id=mock_llm.id, ephemeral=True)
        kept = service.create_conversation(llm_id=mock_llm.id)
        pasted = pasted_dir / "paste-left.png"
        pasted.write_bytes(b"png")
        test_db_session.add(
            Message(conversation_id=left_over.id, sender="user", content=_marker(pasted))
        )
        test_db_session.commit()
        for conversation in (left_over, kept):
            await checkpointer.aput(
                _thread(conversation.id), empty_checkpoint(), {"source": "input", "step": 0}, {}
            )

        purged = await purge_ephemeral_conversations(test_db_session, checkpointer)

        assert purged >= 1
        assert test_db_session.get(Conversation, left_over.id) is None
        assert test_db_session.get(Conversation, kept.id) is not None
        assert await checkpointer.aget_tuple(_thread(left_over.id)) is None
        assert await checkpointer.aget_tuple(_thread(kept.id)) is not None
        assert not pasted.exists()

    async def test_orphan_threads_are_purged_and_a_living_conversation_keeps_its_own(
        self, checkpointer_tables, test_db_session, mock_llm, pg_test_cluster
    ):
        """A crash between the delete of the rows and the delete of the
        thread leaves a thread no conversation owns: the boot sweep removes
        it (its id may be reused by a later conversation)."""
        from src.agents.checkpoint import open_checkpointer

        repo = ConversationRepository(test_db_session)
        living = repo.create_conversation(llm_id=mock_llm.id)
        gone = repo.create_conversation(llm_id=mock_llm.id, ephemeral=True)
        test_db_session.commit()
        gone_id = gone.id
        async with open_checkpointer(pg_test_cluster.psycopg_url) as saver:
            for conversation_id in (living.id, gone_id):
                await saver.aput(
                    _thread(conversation_id),
                    empty_checkpoint(),
                    {"source": "input", "step": 0},
                    {},
                )
            # The crash: the rows went, the thread stayed.
            test_db_session.delete(gone)
            test_db_session.commit()
            # A thread whose id is no conversation id at all is not ours.
            foreign = f"not-a-conversation-{uuid.uuid4()}"
            await saver.aput(
                _thread(foreign), empty_checkpoint(), {"source": "input", "step": 0}, {}
            )
            try:
                purged = await purge_orphan_threads(test_db_session, saver)

                assert purged >= 1
                assert await saver.aget_tuple(_thread(gone_id)) is None
                assert await saver.aget_tuple(_thread(living.id)) is not None
                assert await saver.aget_tuple(_thread(foreign)) is not None
            finally:
                await saver.adelete_thread(str(living.id))
                await saver.adelete_thread(foreign)

    async def test_the_orphan_query_reads_the_three_checkpointer_tables(
        self, checkpointer_tables, test_db_session
    ):
        # The tables adelete_thread clears: a thread with only pending writes
        # or blobs left is an orphan too.
        tables = {
            row[0]
            for row in test_db_session.execute(
                text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
            )
        }
        assert {"checkpoints", "checkpoint_writes", "checkpoint_blobs"} <= tables

    async def test_start_and_stop_sweeps_log_at_info(
        self, test_db_session, mock_llm, monkeypatch, caplog
    ):
        import logging

        from src.domains.conversations import ephemeral

        ConversationRepository(test_db_session).create_conversation(
            llm_id=mock_llm.id, ephemeral=True
        )
        test_db_session.commit()

        async def no_orphans(db, checkpointer):
            return 0

        monkeypatch.setattr(ephemeral, "purge_orphan_threads", no_orphans)
        monkeypatch.setattr(ephemeral, "SessionLocal", lambda: _Unclosable(test_db_session))

        with caplog.at_level(logging.INFO):
            await ephemeral.sweep_at_start(InMemorySaver())
            ConversationRepository(test_db_session).create_conversation(
                llm_id=mock_llm.id, ephemeral=True
            )
            test_db_session.commit()
            await ephemeral.purge_at_shutdown(InMemorySaver())

        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        assert any(m.startswith("Ephemeral conversations purged at startup") for m in messages)
        assert any(m.startswith("Ephemeral conversations purged at shutdown") for m in messages)
        assert all(m.isascii() for m in messages)
        assert (
            test_db_session.query(Conversation).filter(Conversation.ephemeral.is_(True)).count()
            == 0
        )


class _Unclosable:
    """The test session, handed out as a session the sweep closes."""

    def __init__(self, session):
        self._session = session

    def __getattr__(self, name):
        return getattr(self._session, name)

    def close(self):
        pass
