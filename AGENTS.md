# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `src/`: `src/data/` for dataset parsing and preprocessing, `src/retrieval/` and `src/knowledge_base/` for KB search and indexing, `src/reasoning/` for the CoPE chain, `src/rag_pipeline/` for orchestration, `src/baselines/` for comparison models, and `src/evaluation/` for metrics. Entry-point scripts are in `scripts/`, YAML configs in `configs/`, the Streamlit app in `app/`, and automated tests in `tests/`. Keep raw data under `data/raw/`, processed artifacts under `data/processed/`, and generated reports or predictions under `outputs/`.

## Build, Test, and Development Commands
Use the existing `uv` workflow with Python 3.12.

- `make setup`: install the spaCy English model.
- `make test`: run the pytest suite in `tests/`.
- `make lint`: run Ruff checks and formatting verification.
- `make format`: apply Ruff formatting to `src/`, `scripts/`, and `tests/`.
- `make kb-build`: start Qdrant with Docker and build the knowledge base.
- `make rag-xpr-run`: run the main inference pipeline with `configs/rag_xpr_config.yaml`.
- `make evaluate`: generate evaluation reports in `outputs/reports/`.

For one-off commands, follow the repository pattern: `uv run --no-project --python 3.12 --with-requirements requirements.txt python scripts/<name>.py ...`.

## Coding Style & Naming Conventions
Follow Ruff defaults with a 100-character line limit (`pyproject.toml`). Use 4-space indentation, `snake_case` for functions, variables, and modules, and `PascalCase` for classes such as `RAGXPRPipeline`. Prefer small, focused modules and keep config-driven behavior in `configs/*.yaml` instead of hardcoding paths or model settings. Add new scripts with verb-first names like `build_kb.py` or `run_rag_xpr.py`.

## Testing Guidelines
Pytest is configured to discover `tests/test_*.py`. Add new tests under `tests/`, mirror the module under test where practical, and name cases by behavior, for example `test_extract_returns_top_k`. Cover both normal flows and empty-input or failure paths, especially for data loaders, retrieval, and evaluation code. Run `make test` before opening a PR.

## Commit & Pull Request Guidelines
Recent history mixes `Refactor WIP` with clearer messages like `fix: resolve flake8 lint errors...`; prefer the clearer style. Write short, imperative commit subjects with an optional conventional prefix such as `fix:`, `feat:`, or `refactor:`. PRs should summarize the change, list the commands you ran (`make test`, `make lint`, relevant scripts), note any config or dataset assumptions, and include screenshots only when updating the Streamlit demo.

## Data & Configuration Tips
Do not commit secrets or provider keys. Use environment variables such as `LLM_API_KEY`, `LLM_MODEL_NAME`, and `QDRANT_URL`. Treat `data/raw/`, `outputs/`, and local Qdrant state as generated or environment-specific unless a tracked sample is intentionally required.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
