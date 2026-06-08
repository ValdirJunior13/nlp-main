
import os
import shutil
import subprocess
from pathlib import Path

VERSIONS = ["3.10", "3.11"]
ENV_PREFIX = ".compat.venv.py{}{}"
SCRIPT_DIR = Path(__file__).parent
PROJ_DIR = SCRIPT_DIR.joinpath("..").resolve()


def prepare_envinroment(version: str, strategy: str) -> str:
    venv_path = str(
        PROJ_DIR.joinpath(ENV_PREFIX.format(version.replace(".", ""), strategy))
    )
    for command in [
        ["uv", "venv", venv_path, "--python", version, "--seed"],
        [f"{venv_path}/bin/python", "-m", "pip", "install", "--upgrade", "pip", "uv"],
    ]:
        subprocess.run(command, check=True)

    return venv_path


def install_lib(venv: str, strategy: str = "max"):
    command = [
        f"{venv}/bin/python",
        "-m",
        "uv",
        "pip",
        "install",
    ]

    if strategy == "max":
        command.append("--upgrade")

    command.append(".[all,dev]")

    req = Path("requirements/")
    back_req = Path(f"{venv}/.bak.requirements")
    assert not back_req.exists()
    if strategy == "min":
        # Create backup of requirements
        shutil.copytree(req, back_req, dirs_exist_ok=True)
        assert back_req.exists()
        assert len(list(back_req.rglob("*.txt"))) > 0

        # Update requirements with rules
        for p in req.rglob("*.txt"):
            lines = p.read_text().split("\n")
            for i, line in enumerate(lines):
                # Skip comments
                if not line.startswith("#"):
                    lines[i] = line.split(",")[0].replace(">=", "~=")
            p.write_text("\n".join(lines))

    # Run subprocess
    try:
        subprocess.run(command, check=True)
    finally:
        # Maybe update requirements
        if back_req.exists():
            shutil.copytree(back_req, req, dirs_exist_ok=True)


def run_pytest(venv: str):
    command = [f"{venv}/bin/python", "-m", "pytest", "-x"]
    env = dict(TEST_EXPENSIVE_AIBOX_NLP="1")
    env.update(os.environ)
    subprocess.run(command, check=True, env=env)


def run_examples(venv: str):
    bin = f"{venv}/bin/"
    for command in [
        [f"{bin}/python", "-m", "uv", "pip", "install", "papermill", "jupyterlab"],
        *[
            [f"{bin}/papermill", f, "out.ipynb"]
            for f in list(PROJ_DIR.joinpath("examples").glob("*.ipynb"))
        ],
    ]:
        subprocess.run(command, check=True)

    # Cleanup
    PROJ_DIR.joinpath("out.ipynb").unlink(missing_ok=True)


def cleanup_environments():
    glob = f"{ENV_PREFIX.format('', '')}*/"
    for p in PROJ_DIR.glob(glob):
        if not p.is_dir():
            continue

        assert ".compat.venv.py" in p.name
        shutil.rmtree(p)
        p.unlink(missing_ok=True)


def main():
    # Maybe some environments where not cleanup
    cleanup_environments()

    for v in VERSIONS:
        for strategy in ["max", "min"]:
            # Prepare environment
            venv = prepare_envinroment(v, strategy)

            # Install packages
            # We don't use try-except-finally
            #   to allow for manual inspection of
            #   venv in case of failure
            install_lib(venv, strategy)

            # Run pytest
            run_pytest(venv)

            # Run sample codes
            run_examples(venv)

            # Cleanup if no error
            cleanup_environments()


if __name__ == "__main__":
    main()
