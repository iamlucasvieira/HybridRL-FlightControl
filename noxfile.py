import tempfile
from typing import Any

import nox
from nox.sessions import Session


@nox.session(name="lint")
def lint(session):
    session.run("poetry", "install", external=True)
    session.run("poetry", "run", "mypy", "src")


@nox.session()
def tests(session: Session) -> None:
    """Run the test suite."""
    session.run("poetry", "install", external=True)
    session.run("poetry", "run", "pytest", external=True)


@nox.session(name="pre-commit")
def pre_commit(session: Session) -> None:
    """Run pre-commit."""
    args = session.posargs or [
        "run",
        "--all-files",
        "--hook-stage=manual",
        "--show-diff-on-failure",
    ]
    session.install(
        "bandit",
        "black",
        "darglint",
        "flake8",
        "flake8-bugbear",
        "flake8-docstrings",
        "flake8-rst-docstrings",
        "isort",
        "pep8-naming",
        "pre-commit",
        "pre-commit-hooks",
        "pyupgrade",
    )
    session.run("pre-commit", *args)
