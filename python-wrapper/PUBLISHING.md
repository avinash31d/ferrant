# Publishing `liteagent`

End users install the prebuilt native wheel with:

```bash
pip install liteagent
```

The exact `liteagent` project name was unregistered on PyPI when this package
was prepared. The first release must claim it from this repository:

1. Create a pending trusted publisher for project `liteagent` on PyPI.
2. Set its repository owner/name, workflow to `python-wheels.yml`, and
   environment to `pypi`.
3. Push a tag matching the package version, for example `python-v0.1.0`.

The workflow builds ABI3 wheels on Linux, macOS, and Windows and publishes them
through PyPI trusted publishing. No long-lived PyPI token is stored in GitHub.
Increment the version in both `pyproject.toml` and `Cargo.toml` before later
releases.
