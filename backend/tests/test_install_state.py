"""``GET /erudi/startup/install_state``: does this database hold the user's data?

The renderer asks it once, at the first launch of a version that has never
recorded a "last seen version" (#572): an install that already holds a
conversation, a knowledge base or a downloaded model was updated from an earlier version and is
told what changed; a fresh one (or one after "Clear all data") is not. The
catalog seeds ``llms`` rows on every boot (``local = 0``), so only a
downloaded model (``local = 1``) counts. The route only reads: unlike
``/startup/welcome-popup`` it never marks anything.
"""

import pytest

from src.entities.Conversation import Conversation
from src.entities.KnowledgeBase import KnowledgeBase
from src.entities.Llm import Llm
from src.entities.StartupVariables import StartupVariables

pytestmark = pytest.mark.integration


def _llm(db, *, local, name="Model"):
    llm = Llm(name=name, local=local, link=f"org/{name}", type="qwen")
    db.add(llm)
    db.commit()
    return llm


def test_a_fresh_database_holds_no_user_data(client):
    response = client.get("/erudi/startup/install_state")

    assert response.status_code == 200
    assert response.json() == {"has_user_data": False}


def test_catalog_and_in_flight_rows_are_not_user_data(client, test_db_session):
    # Seeded on every boot, and a download that has not finished.
    _llm(test_db_session, local=0, name="Catalog")
    _llm(test_db_session, local=2, name="Downloading")

    assert client.get("/erudi/startup/install_state").json() == {"has_user_data": False}


def test_a_downloaded_model_is_user_data(client, test_db_session):
    _llm(test_db_session, local=1, name="Downloaded")

    assert client.get("/erudi/startup/install_state").json() == {"has_user_data": True}


def test_a_conversation_is_user_data(client, test_db_session):
    # The model may have been deleted since: the conversation still counts.
    llm = _llm(test_db_session, local=0, name="Gone")
    test_db_session.add(Conversation(llm_id=llm.id, name="Old chat"))
    test_db_session.commit()

    assert client.get("/erudi/startup/install_state").json() == {"has_user_data": True}


def test_a_knowledge_base_is_user_data(client, test_db_session):
    # A user who only built a knowledge base still updated an install in use.
    test_db_session.add(KnowledgeBase())
    test_db_session.commit()

    assert client.get("/erudi/startup/install_state").json() == {"has_user_data": True}


def test_it_never_touches_the_welcome_flag(client, test_db_session):
    # Reading /startup/welcome-popup marks it; this route must not, or the
    # welcome dialog of a fresh install would be skipped.
    client.get("/erudi/startup/install_state")

    assert test_db_session.query(StartupVariables).count() == 0
    assert client.get("/erudi/startup/welcome-popup").json() == {"has_already_displayed": False}


def test_a_database_failure_is_a_500(client):
    from unittest.mock import patch

    from src.domains.startup.repository import Startup_Variables_Repository

    with patch.object(
        Startup_Variables_Repository, "has_user_data", side_effect=RuntimeError("db down")
    ):
        response = client.get("/erudi/startup/install_state")

    assert response.status_code == 500
