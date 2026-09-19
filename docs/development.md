# Development

`main` is the canonical source for releases. Keep the historical research
checkout separate. Bring fixes and research changes here as small commits after
reviewing their diff and running the relevant tests. Avoid copying whole research
branches into the release repository.

The repository is a uv workspace. Its ten packages share the `neuro_co` namespace
and can be installed independently. The root is not an installable distribution.
Keep generic environment/state interfaces, models, algorithms, and training in
`neuro-co-core`. Concrete environments, their states and generators, problem
concepts, and classical solver adapters belong in `neuro-co-problems`. Register
environments through `neuro_co.core.env_registry`; use package APIs instead of
duplicating implementations.

## Environment providers

A package can expose an environment constructor in its `pyproject.toml`:

```toml
[project.entry-points."neuro_co.envs"]
"torch.my_problem" = "my_package.env:MyEnv"
```

Use `jax.my_problem` for a JAX provider. The registry discovers installed entries
without core importing their implementations. For registration within a process:

```python
from my_package.env import MyEnv
from neuro_co.core.env_registry import make_env, register_env

register_env("my_problem", MyEnv, backend="torch")
env = make_env("my_problem", size=20)
```

Imports under `neuro_co.core.envs` are compatibility aliases and require
`neuro-co-problems`; new code should use `neuro_co.problems.<problem>.env`.

## Local checks

Install uv, Python 3.11 or 3.12, and Make, then run:

```bash
make setup
make check
```

`make setup` uses Python 3.12 by default. Set `PYTHON=3.11` to use 3.11. It installs
the locked development dependencies and portable backend extras used by CI.
CUDA Mamba kernels and hardware energy tests require separate setup.

`make check` checks versions, lint, formatting, types, tests, strict documentation,
and distribution builds. Individual targets are `versions`, `lint`, `typecheck`,
`test`, `docs`, and `build`. Use `make format` to apply formatter and lint fixes.
On systems without Make, run the commands in the Makefile directly.

Commit `uv.lock`. Use frozen installs for ordinary development; change dependency
constraints deliberately, run `uv lock`, and review the lockfile diff. Commands
use `uv run --no-sync` so checks do not change the installed environment.

Keep tests focused on library behavior and regressions. Study-specific recipes,
launchers, and experiment campaigns belong in the research checkout.
Test changed behavior in its owning package. For changes shared by Torch and JAX,
run the backend parity tests as well as the affected training tests. The default
suite excludes tests marked `gpu` or `slow`; run those on suitable hardware when
the change needs them. Passing CPU tests does not validate CUDA or energy readings.

## Git workflow

Keep only `main` on GitHub. Short local branches are optional:

```bash
git switch main
git pull --ff-only
git switch -c feat/small-fix
# Edit, review the diff, run checks, and commit.
git switch main
git merge --ff-only feat/small-fix
git push origin main
git branch -d feat/small-fix
```

Push `main` explicitly. Do not use `git push --all` or `git push --mirror`.
Keep checkpoints, datasets, experiment outputs, and local environments out of Git.

## Versions and releases

Packages use one shared version. Check it or set the next stable version with:

```bash
uv run --no-sync python scripts/version.py --check
uv run --no-sync python scripts/version.py 0.2.1
make check
```

The helper updates package metadata, existing runtime version strings, citation,
internal dependency bounds, and `uv.lock`. It accepts stable `X.Y.Z` versions only
and restores those files if locking fails. Before 1.0, internal dependencies stay
within one minor release; afterwards, they stay within one major release.
Review and commit the result before tagging a release. Tags identify releases
without adding branches.

For reproducible experiments, record the commit, lockfile, problem parameters,
instance data, seed, hardware, and backend. Use separate output roots for devices
and frameworks. A fixed seed does not guarantee identical results across hardware
or library versions. Adapt the paths and resources in `scripts/oar` to the local
allocation before submitting with `oarsub`.
