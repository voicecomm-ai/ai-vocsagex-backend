#!/bin/bash

set -euo pipefail

echo "[entrypoint] current environment: ${ENV}"

CONFIG_PATH="./config/config.json"

echo "Check whether the 'config.json' exists:"
if [ -e "$CONFIG_PATH" ]; then
    echo "$CONFIG_PATH exists, deleting..."
    rm -f "$CONFIG_PATH"
else
    echo "$CONFIG_PATH does not exist."
fi

case "$ENV" in
    dev)
        cp ./config/config-dev.json "$CONFIG_PATH"
        ;;
    qa)
        cp ./config/config-qa.json "$CONFIG_PATH"
        ;;
    prod)
        cp ./config/config-prod.json "$CONFIG_PATH"
        ;;
    *)
        echo "[entrypoint] Unknown envionment variables: ENV=$ENV,  use 'dev' by default."
        cp ./config/config-dev.json "$CONFIG_PATH"
        ;;
esac

echo "Succeed to set up 'config.json'."
