# User Settings

Persisted user preferences exposed over the `/erudi/user_settings` routes: the global
web-search default, the interface language, the automatic-update preference, the
inference backend and the default reasoning effort.

`default_reasoning_effort` (`none`, `low`, `medium`, `high`, `xhigh`, `medium` by default)
is what a NEW conversation copies at creation and what an arena turn runs at; an existing
conversation keeps the level it copied. See
[Conversations](../guides/conversations.md#how-much-the-model-may-think-reasoning-effort).

`inference_backend` (`auto` or `cpu`) is the one setting the backend reads outside a
request: the FastAPI lifespan reads it once per boot, after the migrations, and swaps
`CUDA_Engine` for `CPU_Engine` when it says `cpu`. Changing it therefore takes effect only
after a backend restart, which the frontend triggers when the user saves. The value is
inert everywhere but the CUDA leg, so the Settings page only shows the control on a
machine running the CUDA engine, or one already pinned back to the processor from it. See
[Hardware Detection](../guides/hardware.md).

::: src.domains.user_settings.endpoints

::: src.domains.user_settings.repository

::: src.domains.user_settings.schemas
