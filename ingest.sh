#!/bin/bash
set -e

echo "starting ingestion ..."

# Check if directory argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 directory"
    exit 1
fi

# Check if provided argument is a directory
if [ ! -d "$1" ]; then
    echo "$1 is not a directory"
    exit 1
fi

# Iterate over each directory
for dir in "$1"/*; do
    # Check if the item is a directory
        echo "$dir"
        make ingest $dir
done