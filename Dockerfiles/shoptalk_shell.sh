#!/usr/bin/env bash
set -euo pipefail

# Manage the ShopTalk Docker development container.
#
# Usage:
#   ./shoptalk_shell.sh            # start container if needed, then open shell
#   ./shoptalk_shell.sh shell      # same as above
#   ./shoptalk_shell.sh start      # build/start container in background
#   ./shoptalk_shell.sh stop       # stop container, keep it available
#   ./shoptalk_shell.sh down       # stop/remove Compose container/network
#   ./shoptalk_shell.sh restart    # recreate the container
#   ./shoptalk_shell.sh status     # show Compose service status
#   ./shoptalk_shell.sh logs       # stream service logs


# Directory containing this script: project-root/Dockerfiles
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yaml"
SERVICE_NAME="shoptalk-dev"

# Pass the host user's UID/GID into Docker Compose.
# docker-entrypoint.sh uses these at container startup to create/reuse a
# matching non-root user inside the container.
export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"
export USER_NAME="${USER_NAME:-shoptalk}"

## General Purpose Compose command ##
# Build the image if needed and start the dev container if needed.
compose() {
  docker compose -f "${COMPOSE_FILE}" "$@"
}
###############################

start_container() {
  warn_if_missing_openai_key
  compose up -d --build "${SERVICE_NAME}" 
}

##########################

warn_if_missing_openai_key() {
  local secret_file="${SCRIPT_DIR}/.env-secret"

  if [[ ! -f "${secret_file}" ]]; then
    cat >&2 <<EOF_WARNING
Warning: ${secret_file} does not exist.

OpenAI-backed app/eval features will not work until you create it.
Create it with:

  cp "${SCRIPT_DIR}/.env-secret-example" "${secret_file}"

Then edit ${secret_file} and set OPENAI_API_KEY.
EOF_WARNING
    return 0
  fi

  if ! grep -Eq '^OPENAI_API_KEY="?[^".][^"]*"?$' "${secret_file}"; then
    cat >&2 <<EOF_WARNING
Warning: ${secret_file} does not appear to contain a usable OPENAI_API_KEY.

OpenAI-backed app/eval features may fail.
Edit ${secret_file} and set:

  OPENAI_API_KEY="your_api_key_here"
EOF_WARNING
  fi
}

####################################

open_shell() {
  start_container

  compose exec \
    --user "${HOST_UID}:${HOST_GID}" \
    -e HOME="/home/${USER_NAME}" \
    -e PS1="(shoptalk) \u@shoptalk:\w\$ " \
    "${SERVICE_NAME}" \
    bash --noprofile --norc -i
}

###############################

usage() {
  cat <<EOF_USAGE
Usage: $0 [shell|start|stop|down|restart|status|logs]

Commands:
  shell     Start the dev container if needed, then open an interactive shell.
            This is the default when no command is supplied.
  start     Build the image if needed and start the dev container in the background.
  stop      Stop the dev container but keep it available for restart.
  down      Stop and remove the Compose container/network. The image is not removed.
  restart   Recreate the dev container.
  status    Show Compose service status.
  logs      Follow logs for the dev container.
EOF_USAGE
}

######################################

cmd="${1:-shell}"

case "${cmd}" in
  shell)
    open_shell
    ;;

  start)
    start_container
    echo "ShopTalk container started."
    ;;

  stop)
    compose stop "${SERVICE_NAME}"
    ;;

  down)
    compose down
    ;;

  restart)
    compose down
    start_container
    echo "ShopTalk container restarted."
    ;;

  status)
    compose ps
    ;;

  logs)
    compose logs -f "${SERVICE_NAME}"
    ;;

  -h|--help|help)
    usage
    ;;

  *)
    echo "Unknown command: ${cmd}" >&2
    usage >&2
    exit 2
    ;;
esac

