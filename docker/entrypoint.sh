#!/usr/bin/env bash
# Install the mounted repo into the image's venv, then run the command. Dependencies are
# already in the image, so this only installs the project itself (plus anything added
# to uv.lock since the image was built; rebuild the image when that happens).
set -euo pipefail
uv sync --no-default-groups --group gpu --group dev --quiet
exec "$@"
