# Contributing to RegimeForge

Thanks for your interest in improving RegimeForge, a regime-aware RL trading research workbench. The Python package is `regime_lens`; `RegimeForge` is the public project name used in documentation and community templates.

## Before you start

- Use the existing `statshell` environment when possible.
- Keep changes focused and reproducible.
- Avoid introducing new dependencies unless they clearly improve the project.

## Workflow

1. Open an issue or clearly describe the change.
2. Make the smallest useful change.
3. Run the relevant tests or smoke checks.
4. Open a pull request with a short summary and validation notes.

## Code standards

- Prefer readable, typed, and deterministic code.
- Match the existing module structure and naming in `backend/regime_lens`.
- Keep public interfaces stable unless the change is intentional.

## Verification

- Run the narrowest test or smoke check that covers the change.
- Include the exact command and outcome in the pull request description.

## Pull requests

- Use a focused title.
- Summarize the user-facing impact and any risks.
- Attach screenshots or logs when UI or experiment output changes.
