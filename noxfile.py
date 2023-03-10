import nox
from nox.sessions import Session


def _setup(session: Session) -> None:
    """Install `av2` into a virtual environment.
    Args:
        session: `nox` session.
    """
    session.run(
        'conda',
        'env',
        'update',
        '--prefix',
        session.virtualenv.location,
        '--file',
        "environment.yml",
        # options
        silent=False)


@nox.session(name='lint')
def lint(session):
    _setup(session)
    session.run('mypy', "src")


@nox.session(name='test')
def test(session):
    _setup(session)
    session.run('pytest', "tests")
