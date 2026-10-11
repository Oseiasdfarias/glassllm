# Releasing glassllm

The package is published to PyPI from GitHub Actions using PyPI Trusted
Publishing. No PyPI API token is stored in GitHub.

## One-time PyPI setup

On PyPI, open the `glassllm` project and configure a GitHub Trusted Publisher
with these values. For an uncreated project, add a [pending publisher from your
PyPI account's publishing settings](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).
For an existing project, use its [Publishing settings](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).

| Field | Value |
| --- | --- |
| Owner | `Oseiasdfarias` |
| Repository | `glassllm` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

If the project has not been created yet, configure a pending publisher for
`glassllm` from PyPI's publishing settings. The first successful tagged
workflow run creates the project and publishes its distributions. If PyPI
reports that the name is already in use, choose an available distribution
name before making a release. A pending publisher does not reserve the name
until it is used.

In GitHub, create an environment named `pypi` under **Settings → Environments**.
Add required reviewers if releases should wait for maintainer approval. The
workflow grants the PyPI job only the `id-token: write` permission it needs for
OIDC publishing.

## Publish a version

1. Update `version` in `pyproject.toml` and commit the release on `main`.
2. Confirm the test workflow passes on Python 3.10 and 3.13.
3. Create and push a tag that matches the package version, for example:

   ```sh
   git tag glassllm-v0.1.0
   git push origin glassllm-v0.1.0
   ```

4. The publish workflow reruns the Python test suite, checks the tag against
   `pyproject.toml`, builds the wheel and source distribution, then publishes
   those exact artifacts to PyPI through Trusted Publishing.

PyPI does not allow replacing files for an already published version. If a
release fails after publishing, increment the package version before trying
again.
