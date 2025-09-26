# Repository Guidelines

## Project Structure & Module Organization
- `web_app.py`: Streamlit UI for enrollment and verification.
- `voice_enrollment_with_stt.py`: Audio capture, Google STT streaming, silence trimming, and speaker verification helpers.
- `main.py`: Simple CLI demo for enrolling/verifying locally.
- `enrolled/`: Generated audio and index files (per-user `*_enrolled_files.txt`).
- `pyproject.toml` / `uv.lock`: Python 3.11+ project and locked dependencies (managed with `uv`).

## Build, Test, and Development Commands
- Install deps (creates managed venv): `uv sync`
- Run Streamlit app: `uv run streamlit run web_app.py`
- Run CLI demo: `uv run python main.py`
- Pin new deps: update `pyproject.toml` then `uv sync`

## Coding Style & Naming Conventions
- Python: PEP 8, 4-space indentation, type-friendly code where practical.
- Names: `snake_case` for functions/vars, `PascalCase` for classes; descriptive filenames (e.g., `voice_enrollment_with_stt.py`).
- Modules should avoid hardcoding paths; prefer `enrolled/` as the default data root.
- Keep functions small and focused; add docstrings to public helpers.

## Testing Guidelines
- No formal tests yet. If adding:
  - Place tests in `tests/`, name files `test_*.py`.
  - Use `pytest`; run with: `uv run pytest -q`.
  - Store tiny audio fixtures under `tests/data/`; avoid large files in Git.

## Commit & Pull Request Guidelines
- Commits: imperative subject, <= 72 chars; include a scope when helpful (e.g., `ui:`, `audio:`, `stt:`, `verify:`). Example: `stt: trim trailing silence in saved WAVs`.
- PRs: clear description, linked issues, reproduction steps, and before/after notes. Include a short clip or screenshot of the Streamlit UI when relevant.

## Security & Configuration Tips
- Credentials: set `GOOGLE_APPLICATION_CREDENTIALS` via environment or `.env`; do not commit keys.
  - Example `.env`:
    - `GOOGLE_APPLICATION_CREDENTIALS=/abs/path/to/gcp-key.json`
- Data hygiene: keep personal audio local. Consider ignoring large WAVs (e.g., `enrolled/*.wav`, `verify.wav`) in Git.
- Quotas/latency: Google STT is networked—handle failures gracefully and avoid tight retry loops.

## Architecture Overview
- Enrollment: pick sentences → record → STT transcript similarity gate → save WAVs under `enrolled/`.
- Verification: record once → compare against enrolled samples with SpeechBrain ECAPA → average score threshold (~0.45) to decide.

