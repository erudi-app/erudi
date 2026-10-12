# User Settings

Persisted user preferences exposed over the `/erudi/user_settings` routes: the global
web-search default, the interface language, the automatic-update preference, the
inference backend, the default reasoning effort and the default compaction threshold.

`default_reasoning_effort` (`none`, `low`, `medium`, `high`, `xhigh`, `medium` by default)
is what a NEW conversation copies at creation and what an arena turn runs at; an existing
conversation keeps the level it copied. See
[Conversations](../guides/conversations.md#how-much-the-model-may-think-reasoning-effort).

`default_compaction_threshold` (a whole percent from 50 to 90 in steps of 5, 80 by
default) is the share of the working context window at which a NEW conversation compacts;
the conversation copies it at creation and keeps it, so a later change never touches an
existing conversation. Arena turns never compact and do not read it. A `PUT` carrying only
this field is a valid update; a value off the scale is a 422. See
[Conversations](../guides/conversations.md#how-conversation-state-is-kept).

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
