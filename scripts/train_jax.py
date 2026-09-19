"""Use the shared training CLI with the JAX backend."""

import sys

from neuro_co.cli.main import main


def arguments(argv: list[str]) -> list[str]:
    names = {argument.split("=", 1)[0] for argument in argv}
    if "--resume" in names:
        raise SystemExit(
            "This shared-protocol entry point does not resume runs. For legacy runs only, use: "
            "python -m neuro_co.core.jax_backend.train --resume PATH"
        )
    removed = names & {"--log-every", "--save-every"}
    if removed:
        raise SystemExit(
            f"{', '.join(sorted(removed))} is no longer used; progress is automatic and checkpoints are saved each epoch"
        )
    if "--steps" in names and names & {"--epochs", "--steps-per-epoch"}:
        raise SystemExit("Use either --steps or --epochs/--steps-per-epoch, not both")
    aliases = {"--output": "--out-dir", "--learning-rate": "--lr"}
    result = ["train", "--backend", "jax"]
    if "--problem" not in names:
        result += ["--problem", "cvrp"]
    for argument in argv:
        name, equal, value = argument.partition("=")
        if name == "--steps":
            result += ["--epochs", "1"]
            name = "--steps-per-epoch"
        result.append(aliases.get(name, name) + (equal + value if equal else ""))
    return result


if __name__ == "__main__":
    main(arguments(sys.argv[1:]))
