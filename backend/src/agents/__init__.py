"""LangChain/LangGraph agent layer for Erudi.

A thin transverse layer (consumed by ``domains/conversations`` and
``domains/arena`` services) that wraps the engine behind LangChain primitives:

  - ``model_factory`` — build a ``ChatOpenAI`` pointed at the local engine server
    and the engine-level serialization/idle-marker guard.
  - ``runner`` — ``AgentRunner``: one ``create_agent`` per turn, streamed as
    structured events the services frame as NDJSON.
  - ``curated_turns`` — the app's own sentences for a turn it could not answer
    normally, each with the kind and numbers the interface translates it from.
  - ``middleware`` — per-turn request-time middleware (KB merge, image/tool-result
    hygiene), imported lazily by the runner so LangChain stays out of boot (#160).
  - ``prompts`` — size-adaptive system prompt construction.
  - ``checkpoint`` — PostgreSQL checkpointer wiring (AsyncPostgresSaver on the `erudi` database).
"""
