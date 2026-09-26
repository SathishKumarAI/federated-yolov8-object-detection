# `Research_docs/installations/`

Notes and environment records from the project's research phase. Nothing here is
installed by the pipeline, by `my-project`, or by CI. The manifests that are actually
used are `my-project/pyproject.toml` and `pipeline/requirements.txt`.

## `requirements_history/` was removed on 2026-09-26

It held four files:

```
requirements_20240621155143_new.txt
requirements_20240621155143_old.txt
requirements_20240621161334_old.txt
requirements_20240621161625_old.txt
```

All four had the **same md5** (`4611da35399aaf8da2fd6ae9e2009603`) — one 409-line
`pip freeze` of a conda environment from 2024-06-21, stored four times under names that
suggest a history that does not exist. 57 of those lines look like

```
absl-py @ file:///C:/b/abs_5babsu7y5x/croot/absl-py_1666362945682/work
```

— paths on one machine, two years ago. The file cannot be installed anywhere, by anyone.

**They were producing essentially all of the repository's Dependabot alerts**: 933 open,
8 critical and 47 high among the first hundred, none of them in code this project runs.
`my-project/pyproject.toml` and `pipeline/requirements.txt` had **zero** between them. An
alert list that size is an alert list nobody reads, which is the real cost — a genuine
advisory against `torch` or `flwr` would have landed in it unseen.

Removing them was backlog item **106 (P1)**, which had already done the measurement and
prescribed `git rm`.

**Nothing is lost.** Git has them:

```bash
git log --oneline --all -- Research_docs/installations/requirements_history/
git show <commit>^:Research_docs/installations/requirements_history/requirements_20240621155143_new.txt
```

## `cuda_test_file/requirements.txt` was deliberately kept

It is a live file — a ~60-package dump including `fedml`, `tensorflow_federated`,
`tritonclient` and `mxnet~=1.6.0`, pinned in 2022. It carries alerts of its own, and
backlog 106 said to keep it and triage it separately rather than fold it into a cleanup
it has nothing to do with.

Nothing installs it either. If the alerts it raises are not wanted, the options are to
delete it the way the duplicates went, or to dismiss its alerts as `not_used` — which is
true, but is a decision about the repository's security surface and so belongs to its
owner, not to a cleanup commit.
