

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


## DVC data

Dataset is tracked with DVC on a DagsHub remote.

### First-time setup

1. Create a free DagsHub account and ask Amos to add you as a collaborator on https://dagshub.com/AmosShehzad/wastewise-ml-collab
2. Get your personal DVC tokens: DagsHub repo page -> Remote -> Data -> DVC
3. Configure your local credentials (NEVER commit these):

\\ash
uv run dvc remote modify storage --local access_key_id <YOUR_TOKEN>
uv run dvc remote modify storage --local secret_access_key <YOUR_TOKEN>
\
### Daily workflow

\\ash
uv run dvc pull     # before you start work (fetch latest data)
uv run dvc push     # ALWAYS before git push (if you changed data)
\
