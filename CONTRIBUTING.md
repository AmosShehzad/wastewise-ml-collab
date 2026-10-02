
# Contributing

## Branch naming

- `main`: production, tagged releases only
- `staging`: release candidate
- `dev`: integration of finished work
- `feat/`: features and pipeline changes (from `dev`)
- `data/ `: dataset updates tracked with DVC (from `dev`)
- `exp/
- 

`: experiments (from `dev`), never merged directly

- `fix/ `: urgent fix to production (from `main`)
  Nobody pushes directly to `dev`, `staging` or `main`. All changes arrive through pull requests.

## Commit messages

We use Conventional Commits, for example:

- `feat: add scaling step`
- `data: remove duplicate rows`
- `exp: try max_depth=8`

## Merge strategy

Pull requests into `dev` are squash-merged.
