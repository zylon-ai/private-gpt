#!/bin/sh

# Check if the FILE_URL environment variable is set
if [ -z "$FILE_URL" ]
then
    echo "Error: FILE_URL environment variable is not set."
    exit 1
fi

wget -O "models/${NAME}" "${FILE_URL}"

# Execute the main container command
exec "$@"