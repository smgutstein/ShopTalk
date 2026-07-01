#!/usr/bin/env bash
set -euo pipefail

# Create or reuse a container user that matches the host user's UID/GID, then
# run the requested command as that user. This keeps files created inside the
# bind-mounted repo from becoming root-owned on the host.

HOST_UID="${HOST_UID:-1000}"
HOST_GID="${HOST_GID:-1000}"
USER_NAME="${USER_NAME:-shoptalk}"
GROUP_NAME="${GROUP_NAME:-shoptalk}"

# Reuse an existing group with the requested GID when possible; otherwise
# create one using GROUP_NAME.
if getent group "${HOST_GID}" >/dev/null; then
    GROUP_NAME="$(getent group "${HOST_GID}" | cut -d: -f1)"
else
    groupadd --gid "${HOST_GID}" "${GROUP_NAME}"
fi

# Reuse an existing user with the requested UID when possible; otherwise create
# one using USER_NAME and the host-matching group.
if getent passwd "${HOST_UID}" >/dev/null; then
    USER_NAME="$(getent passwd "${HOST_UID}" | cut -d: -f1)"
else
    useradd \
        --uid "${HOST_UID}" \
        --gid "${HOST_GID}" \
        --create-home \
        --shell /bin/bash \
        "${USER_NAME}"
fi

# Make HOME match the selected user so tools write caches/configs to a sane
# location instead of trying to use /root.
export HOME="$(getent passwd "${USER_NAME}" | cut -d: -f6)"

# The repo is normally bind-mounted here by Compose. Do not recursively chown
# /workspace; matching UID/GID should already make host files writable, and
# chowning a large repo or dataset mount can be slow and surprising.
mkdir -p /workspace
cd /workspace

exec gosu "${USER_NAME}" "$@"