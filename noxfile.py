import nox
from nox.sessions import Session
from typing import Any
import tempfile


@nox.session(name='lint')
def lint(session):
    session.run("poetry", "install", external=True)
    session.run("poetry", "run", 'mypy', "src")


@nox.session()
def tests(session: Session) -> None:
    """Run the test suite."""
    session.run("poetry", "install", external=True)
    session.run("poetry", "run", "pytest", external=True)
