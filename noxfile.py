import nox

nox.options.sessions = ["format", "lint"]


@nox.session(python="3.10", reuse_venv=True)
def format(session):
    session.install("black")
    session.run("black", ".", *session.posargs)


@nox.session(python="3.10", reuse_venv=True)
def lint(session):
    session.install("flake8")
    session.run("flake8", "constants.py", "ingest.py", "privateGPT.py")
