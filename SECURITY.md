# Security policy

Erudi runs language models on your own machine, and its main promise is that your prompts, documents and conversations never leave it. A security problem in Erudi is therefore a privacy problem for every user, and we treat reports accordingly.

## Reporting a vulnerability

Please do **not** open a public issue for anything you believe is a security vulnerability.

Use GitHub's private reporting instead: **[Report a vulnerability](https://github.com/erudi-app/erudi/security/advisories/new)**. It opens a private thread between you and the maintainers, lets us work on a fix without exposing users, and credits you in the advisory once it is published.

If you cannot use GitHub for some reason, email `contact@erudi.app` with "security" in the subject line.

What helps us act fast: the version of Erudi (Settings shows it, or the release tag), your operating system, a description of the impact, and steps or a proof of concept that reproduce the problem. Logs are welcome, but please review them first — they contain conversation content.

## What to expect

- We acknowledge reports within **5 working days**.
- We aim to confirm the problem and agree on a severity within **two weeks**, and to ship a fix in the next release after that. Critical issues (remote code execution, anything that sends user data off the machine) get a dedicated release.
- We will keep you informed as we go, and we will not publish details before a fix is available unless you agree otherwise.
- Once fixed, the advisory is published on the [security page](https://github.com/erudi-app/erudi/security) with credit to the reporter.

Erudi is a volunteer-run open-source project. There is no bug bounty programme, but every report is read and answered.

## Scope

The network surface of the application — every request it can make, when, and what it exposes locally — is documented with code references on [What leaves your machine](https://www.erudi.app/open-source/desktop/docs/privacy). A report that shows that page to be wrong is a valid security report.

In scope: the Erudi desktop application (the Electron frontend, the Python backend it spawns, the packaged installers) and the release pipeline that produces them.

The backend listens only on `127.0.0.1` and accepts requests only from the packaged renderer; anything that lets another local process or a web page reach it, read another user's data, or execute code is in scope. So is anything that makes the app contact a network endpoint other than the model download it was explicitly asked to perform.

Out of scope: vulnerabilities in the language models themselves (prompt injection, harmful output), in third-party model repositories on Hugging Face, or in dependencies that do not affect Erudi's behaviour. Reports against those are still welcome upstream.

## Supported versions

Only the latest release receives security fixes. Erudi updates itself; if you have disabled updates, please check the [releases page](https://github.com/erudi-app/erudi/releases) before reporting.
