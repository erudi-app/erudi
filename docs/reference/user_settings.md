# User Settings

Persisted user preferences exposed over the `/erudi/user_settings` routes: the global
web-search default, the interface language, the automatic-update preference and the
inference backend.

`inference_backend` (`auto` or `cpu`) is the one setting the backend reads outside a
request: the FastAPI lifespan reads it once per boot, after the migrations, and swaps
`CUDA_Engine` for `CPU_Engine` when it says `cpu`. Changing it therefore takes effect only
after a backend restart, which the frontend triggers when the user saves. See
[Hardware Detection](../guides/hardware.md).

::: src.domains.user_settings.endpoints

::: src.domains.user_settings.repository

::: src.domains.user_settings.schemas
