

## Pre-commit hooks

After cloning, every member must run once:

```bash
uv sync
uv run pre-commit install
```

Hooks enforced on every commit:
- ruff (lint + format)
- nbstripout (strips notebook outputs)
- check-added-large-files (blocks files > 1 MB)
- detect-secrets (blocks new secrets vs .secrets.baseline)
