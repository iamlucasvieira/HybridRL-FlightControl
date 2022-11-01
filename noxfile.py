import nox

@nox.session(name='lint')
def lint(session):
    session.install('flake8', ".", "mypy")
    session.run('mypy', "src")
