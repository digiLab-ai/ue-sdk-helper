# ue-sdk-helper

Utilities for the Uncertainty Engine ecosystem.

## Table of contents
- [ue-sdk-helper](#ue-sdk-helper)
  - [Table of contents](#table-of-contents)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Repository structure](#repository-structure)
  - [Development workflow](#development-workflow)
    - [Branch naming](#branch-naming)
    - [Commits](#commits)
    - [Pull requests](#pull-requests)
      - [PR template (paste this into the PR body)](#pr-template-paste-this-into-the-pr-body)
    - [PR checklist](#pr-checklist)
  - [Testing expectations](#testing-expectations)
  - [Code style](#code-style)
  - [Local dev quickstart](#local-dev-quickstart)

---

## Overview
`ue-sdk-helper` is a small helper library for the Uncertainty Engine stack. It collects frequently used utilities (resource interaction, workflow rappers, plotting helpers, data generators, notebook ergonomics) so downstream projects can depend on one place instead of re-implementing the same snippets.

## Installation
```bash
pip install git+https://github.com/digiLab-ai/ue-sdk-helper.git@main
# or, from source
pip install -e ".[dev]"
```

The project depends on the Uncertainty Engine:
```bash
pip install "uncertainty-engine" matplotlib numpy pandas seaborn
```

## Repository structure
```
ue-sdk-helper/
├─ src/
│  └─ ue_sdk_helper/
│     ├─ __init__.py
│     ├─ plotting.py
│     ├─ workflows.py
│     └─ notebook.py
├─ tests/
│  ├─ test_plotting.py
├─ notebooks/
│  └─ 00_client.ipynb
├─ pyproject.toml
├─ README.md
├─ LICENSE
├─ .gitignore
├─ .pre-commit-config.yaml
└─ Makefile
```

## Development workflow

### Branch naming
Create a short, descriptive branch that starts with your initials (or GitHub handle) and the objective:

- Feature: `dg/feat-dataset-generator`
- Fix: `dg/fix-NaN-handling-stats`
- Chore/CI: `dg/chore-ci-precommit`
- Issue-scoped: `dg/123-fix-bounds-checking` (where 123 is the issue number)

**Tip:** keep branches focused; one logical change per branch.

### Commits
- Conventional, present-tense, and scoped where helpful:
  - `feat(stats): add weighted_mean with tests`
  - `fix(io): handle missing .env gracefully`
  - `docs(readme): add dev workflow`
- Keep commits small and meaningful; include rationale when non-obvious.

### Pull requests
Open a PR early as a draft; convert to “Ready for review” when:
- All tests pass locally
- Lint/format checks pass
- PR description explains **objective**, **approach**, and **trade-offs**
- Added/updated tests demonstrate the change

#### PR template (paste this into the PR body)
```
## Objective
(What problem are we solving? Link issues if applicable.)

## Approach
(What did you implement? Why this design?)

## Tests
(What unit tests were added/updated? Any coverage notes?)

## Breaking changes?
(Yes/No — if yes, describe migration.)

## Screenshots / Notes
(Optional: plots, logs, or snippets useful for reviewers.)
```

### PR checklist
- [ ] Branch name uses `initials/...`
- [ ] New code has unit tests
- [ ] `make format lint test` passes locally
- [ ] Public APIs documented (docstrings / README / examples)
- [ ] No large, unrelated refactors mixed in

## Testing expectations
- **Unit tests** live in `tests/` and must cover new code paths.
- **Coverage target:** ≥ **85%** for changed files (keep simple utilities near 100%).
- Use **pytest**; prefer **property-based** checks for tiny helpers when sensible.
- Plotting functions: test **pure parts** (e.g., data preparation) and smoke-test figure creation (no exceptions).

Example test:
```python
# tests/test_stats.py
import numpy as np
from ue_sdk_helper.stats import weighted_mean

def test_weighted_mean_basic():
    x = np.array([1.0, 2.0, 3.0])
    w = np.array([1.0, 1.0, 2.0])
    assert np.isclose(weighted_mean(x, w), 2.25)

def test_weighted_mean_handles_empty():
    assert np.isnan(weighted_mean(np.array([]), np.array([])))
```

## Code style
- **Black** for formatting
- **Ruff** for linting (`ruff check` + `ruff format` or let Black format)
- **Pre-commit** hooks run Black, Ruff, end-of-file fixes, and trailing whitespace
- Optional: **mypy** if/when we add types


## Local dev quickstart
```bash
# 1) Set up environment
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2) Install pre-commit
pre-commit install

# 3) Run quality gates
make format lint test
# or the raw commands:
# ruff check .
# black --check .
# pytest -q --maxfail=1 --disable-warnings
```

Example `Makefile`:
```make
.PHONY: format lint test all
format:
	black src tests
	ruff format src tests
lint:
	ruff check src tests
test:
	pytest -q --maxfail=1 --disable-warnings
all: format lint test
```
