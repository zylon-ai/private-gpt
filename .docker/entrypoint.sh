#!/bin/bash

# Ensure correct ownership of the local_data directory
chown -R worker /home/worker/app/local_data

# Ensure correct ownership of the models directory if it exists
if [ -d "/home/worker/app/models" ]; then
    chown -R worker /home/worker/app/models
fi

# Execute the command passed to the entrypoint
exec "$@"