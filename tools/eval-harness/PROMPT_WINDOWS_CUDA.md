# Prompt for the coding agent (Windows CUDA machine)

Paste everything below the line into your coding agent, from a terminal opened in the folder where you unzipped `erudi-eval.zip`.

---

You are running a memory and storage evaluation of the **installed** Erudi desktop app (CUDA build) on this Windows machine, with a harness that was written and tested on macOS only. Your job: make it run correctly on Windows, run it, and bring back the results. Answer me in French; keep code and code comments in English. Nothing here is committed anywhere: work in this folder only.

## Context

- The harness is in this folder: `erudi_eval.py`, package `erudi_eval/`, `tests/`, `corpus/`, `docs/`. Read `README.md`, then `docs/SPEC.md` (what must be measured) and `docs/APP_REFERENCE.md` (facts about the packaged app: paths, process tree, HTTP API, lifecycle events) before running anything.
- Windows-specific parts were written from the Erudi source code and have **never run on Windows**: install/data/log paths, process classification (`Erudi.exe` helpers with `--type=`, `backend.exe`, `postgres.exe` under `pginstall\bin`, `llama-server.exe`), the exe version query, `taskkill /PID` quit, NVML / `nvidia-smi` per-process VRAM, the storage walker's logical sizes. Expect to fix some of them.
- The Erudi app must be the **CUDA** build, version 1.1.2, installed per-user (default `%LOCALAPPDATA%\Programs\Erudi\Erudi.exe`). If it is not installed or not 1.1.2, stop and tell me.
- Every run of this prompt uses the flavour **`win-cuda`**: pass `--flavour win-cuda` so the harness refuses to measure if the installed app turns out to be the CPU build (it asks the running app for its `backend_type` and engine). The flavour decides the model (`Qwen/Qwen3-4B-GGUF`), the expected inference process (`llama-server.exe` from `...\artifacts\llama-cpp\cuda\bin`), that per-process VRAM is expected from NVML or `nvidia-smi`, and that `-ngl` and `--threads` are recorded from the child's command line. Its definition is `flavours/win-cuda.json`.

## Hard safety rules

1. **Never kill, stop or signal a process by name or pattern** (no `taskkill /IM`, no `Stop-Process -Name`, no loops over `Get-Process` matches). Only the harness's own quit of the Erudi main process it identified is allowed. If something must be stopped and it is not obviously that process, ask me first.
2. Never delete the Erudi data directory (`%LOCALAPPDATA%\erudi`), existing models, conversations or knowledge bases. Only run with `--cleanup` (which deletes only what the harness created) once a run has succeeded.
3. Check free disk before any run: the default model `Qwen/Qwen3-4B-GGUF` is about 2.5 GB, plus the embedding model, plus 5 GB headroom.
4. Do not read `erudi_db_password`, do not connect to Postgres or to `llama-server` directly, do not print secrets.
5. Make sure no Erudi process is running before a run (the harness refuses otherwise). If one is, ask me to quit the app from its UI.

## Steps

1. Install `uv` if missing (`winget install --id=astral-sh.uv -e` or the official PowerShell installer), then `uv --version`.
2. `uv run erudi_eval.py selftest`. Check that it finds the install, the data root (`%LOCALAPPDATA%\erudi\backend\prod`), the stdout capture log (`%TEMP%\erudi-backend.log`), that the primary metric is `private_working_set`, and that a GPU source (NVML or `nvidia-smi`) is detected. Fix whatever is wrong in the harness code.
3. `uv run --with pytest pytest tests/`. Platform-specific tests may skip; any failure on Windows must be understood and fixed (or explained if it is a macOS-only assumption in the test).
4. Short smoke run first, to validate discovery and metrics on the real app without long phases:
   `uv run erudi_eval.py run --flavour win-cuda --phases preflight,cold_boot,idle_after_boot,ui_tour,quit --idle-seconds 30`
   Then open `results/<run>/report.md` and check, before going further:
   - every Erudi process is attributed to a category (electron_main, electron_renderer, electron_gpu, electron_utility, backend, database); nothing Erudi-related is missing (compare with Task Manager's process tree);
   - no non-Erudi process was counted;
   - the boot timeline has `starting`, the `phase` events and `ready`;
   - the storage breakdown shows the install dir and the data root;
   - after `quit`, no survivor at +30 s.
   Fix the harness until this is right. Keep every fix minimal and note why.
5. **Two full runs, one per profile** (the point is to compare a clean machine with a realistic one):
   - **clean**: reboot Windows, open only the terminal you run the harness from, then
     `uv run erudi_eval.py run --flavour win-cuda --profile clean --cleanup`.
   - **nominal**, right after, without rebooting: open Chrome with 4 tabs (Gmail, Google Docs, a paused YouTube video, GitHub), the chat apps idle, Windows Terminal with 5 coding-agent CLI sessions open and idle for the whole run, then
     `uv run erudi_eval.py run --flavour win-cuda --profile nominal --cleanup`.
   - then `uv run erudi_eval.py compare results/<clean-run> results/<nominal-run>` and keep the output.

   `workloads/clean-windows.json` and `workloads/nominal-windows.json` are **examples**: their rules are `<placeholders>`, and a group whose placeholders are still in place is reported as "not configured" and matches nothing. Copy each one to `workloads/local-clean-windows.json` / `workloads/local-nominal-windows.json` (gitignored) and fill in the real install paths of the browser, the chat apps, the terminal and the agent CLI you are running, then pass it with `--workload` (a run with no `--workload` picks the local file automatically when it exists). **None of these rules has ever run on Windows.** Check the report's Workload section: every group you actually have open must be found with the right process count. If a group is missing or captures the wrong processes, fix its rules (`exe_prefixes`, `exe_basenames`, `argv0_basenames`, `argv0_prefixes`, `cmdline_contains` — the format is documented in the README) and say in your report what you had to change and why. Run `uv run erudi_eval.py selftest` to iterate quickly: it prints the workload check without running anything.

   Note: a run whose declared workload does not match is a warning, not a failure. Do not use `--strict-workload` unless you want the run to refuse to start.

6. In the report of each full run, read the **Flavour and model** section and check:
   - the running app confirmed `win-cuda` (backend_type `cuda`, engine `CUDA_Engine`);
   - the inference process was seen as `llama_server` and the `inference` category contains `llama-server.exe`;
   - `-ngl` is greater than 0 (layers really offloaded to the GPU) and `--threads` is recorded;
   - per-process VRAM is a number, or is explicitly reported as unavailable — under the WDDM driver model NVML often returns `[N/A]`, in which case the per-GPU used memory in the machine context is the only VRAM figure. Say which one you got.
   Also check the **model** line: an already-installed model must be reused ("matched by link" or "matched by name"), never downloaded twice. If a download starts while the model is already installed, that is a bug worth reporting.
7. If a phase fails, read `events.jsonl`, `phases.json` and the copied logs in `results/<run>/logs/`, decide whether it is a harness bug (fix and re-run that phase with `--phases`) or an app behaviour (record it, do not "fix" the app).

## What to bring back

1. A zip of the `results/` folder (all runs, including failed ones).
2. A patch of your harness changes. Before editing anything, copy the untouched folder to `..\erudi-eval-original`; at the end produce `git diff --no-index ..\erudi-eval-original . > windows-fixes.patch` (excluding `results/`, `.venv/`, `__pycache__/`), and list each change with its reason (file, what was wrong on Windows, what you changed).
3. A short report in French:
   - machine: CPU, RAM, GPU model and VRAM, NVIDIA driver version, Windows version, Erudi version and flavour;
   - for both profiles, the conditions printed in the report header (uptime, swap at start, power) and the headline numbers from `report.md`: app overhead (everything except inference) at idle after boot, during chat and after KB ingestion; inference memory (RAM and VRAM) during chat; boot duration; TTFT cold and warm; install size and data root size;
   - the workload verification and any drift warning (a group that changed size or got busy during a run makes its numbers less comparable);
   - anything surprising (processes that survived quit, metrics that could not be read, phases skipped and why);
   - what you could not verify.
