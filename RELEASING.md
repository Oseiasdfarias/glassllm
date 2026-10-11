# Releasing glassllm

The package is published to PyPI from GitHub Actions using PyPI Trusted
Publishing. No PyPI API token is stored in GitHub.

## PyPI publisher

The pending GitHub Trusted Publisher is configured on PyPI with the values
below. On its first successful upload, PyPI creates `glassllm` and converts the
pending publisher into a project publisher. A pending publisher does not reserve
the package name until it is used.

| Field | Value |
| --- | --- |
| Owner | `Oseiasdfarias` |
| Repository | `glassllm` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

The `pypi` environment is referenced by the workflow and by the PyPI publisher.
The workflow grants its publishing job only the `id-token: write` permission
needed for OIDC; no long-lived PyPI API token is stored in GitHub.

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
