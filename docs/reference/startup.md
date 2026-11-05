# Startup

First-run seeding, startup variables and job cleanup served by the `/erudi/startup` routes.

`GET /erudi/startup/install_state` answers `{"has_user_data": bool}`: whether the
database holds at least one conversation or one downloaded model (`llms.local = 1`; the
catalog rows the boot seeds are `local = 0` and do not count). It only reads. The renderer
asks it at the first launch of a version that recorded no last-seen version, to tell an
install updated from an earlier version (shown an "Updated to" dialog) from a fresh one,
or one after "Clear all data" (shown nothing). `GET /erudi/startup/welcome-popup` cannot
answer that question: reading it marks the welcome dialog as shown.

::: src.domains.startup

::: src.domains.startup.endpoints

::: src.domains.startup.repository

::: src.domains.startup.schemas
