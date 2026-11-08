# User Settings

Persisted user preferences exposed over the `/erudi/user_settings` routes: the global
web-search default, the interface language, the three update switches, the
inference backend, the default reasoning effort, the default compaction threshold and the
idle unload delay.

The three update switches are read by the Electron main process, which owns
electron-updater; the renderer reads them here and hands them over at launch and on every
change. `auto_update_enabled` is **Check for updates** (`true` by default): the app asks
the update feed for a new version at launch, then every 4 hours, and shows a banner when
there is one. `auto_download_updates` (**Download updates automatically**) and
`auto_install_updates` (**Install downloaded updates when I quit**) are opt-ins, `false`
by default and `false` after the migration that added them, whatever automatic updates did
before. Both only act while the check is on: with `auto_update_enabled` off they are inert,
and the Settings page disables them. When the renderer cannot read the settings, main
checks, downloads and installs nothing until a later launch reads them. The manual
**Update Erudi** card works whatever the three say. See
[What leaves your machine](../privacy.md).

`default_reasoning_effort` (`none`, `low`, `medium`, `high`, `xhigh`, `medium` by default)
is what a NEW conversation starts at and what an arena panel runs at, on every model that has
no level saved with `/config global`; an existing conversation keeps the level it started
at. See
[Conversations](../guides/conversations.md#how-much-the-model-may-think-reasoning-effort).

`default_compaction_threshold` (a whole percent from 50 to 90 in steps of 5, 80 by
default) is the share of the working context window at which a NEW conversation compacts,
on every model that has no threshold saved with `/config global`; the conversation copies
it at creation and keeps it, so a later change never touches an existing conversation. Arena turns never compact and do not read it. A `PUT` carrying only
this field is a valid update; a value off the scale is a 422. See
[Conversations](../guides/conversations.md#how-conversation-state-is-kept).

`web_search_enabled` is, the same way, the starting switch of a new conversation on a model
without a saved one, and the switch every arena panel follows. The values saved per model
are described in
[Conversations](../guides/conversations.md#conversation-defaults-per-model).

`inference_backend` (`auto` or `cpu`) is the one setting the backend reads outside a
request: the FastAPI lifespan reads it once per boot, after the migrations, and swaps
`CUDA_Engine` for `CPU_Engine` when it says `cpu`. Changing it therefore takes effect only
after a backend restart, which the frontend triggers when the user saves. The value is
inert everywhere but the CUDA leg, so the Settings page only shows the control on a
machine running the CUDA engine, or one already pinned back to the processor from it. See
[Hardware Detection](../guides/hardware.md).

`idle_unload_minutes` (`0`, `5`, `15`, `30` or `60`, `5` by default and `5` after the
migration that added it) is how long the chat model stays loaded once unused; `0` is the
Settings card's **Never**, which keeps the model loaded however long it stays unused. The
lifespan reads it at boot and gives it to the engine that runs (after the CUDA-to-CPU swap),
and a `PUT` carrying it applies it to that engine at once, so no relaunch is needed. Any
other value, a boolean, a float or a numeric string is a 422. Nothing below five minutes is
offered: the memory brake's idle timings are calibrated against a reap at five minutes or
later. With Never the memory brake still frees memory when the computer runs short, and the
header's Unload button still unloads the model. The delay does not apply to the knowledge-base
embedding model, which, once used, stays loaded until the backend stops. See [LLMs](../guides/llms.md#memory-and-the-idle-unload).

::: src.domains.user_settings.endpoints

::: src.domains.user_settings.repository

::: src.domains.user_settings.schemas
