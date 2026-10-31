#!/bin/bash
set -e

# Directory to check for ownership (your mounted volume)
TARGET_DIR="/workspace"

# Get the UID/GID of the mounted directory
if [ -d "$TARGET_DIR" ]; then
    DIR_UID=$(stat -c "%u" "$TARGET_DIR")
    DIR_GID=$(stat -c "%g" "$TARGET_DIR")

    # Only modify if different from current docker user
    CURRENT_UID=$(id -u docker)
    CURRENT_GID=$(id -g docker)

    if [ "$DIR_UID" != "$CURRENT_UID" ] || [ "$DIR_GID" != "$CURRENT_GID" ]; then
        echo "Adjusting docker user UID:GID to $DIR_UID:$DIR_GID"

        # Modify the docker group GID
        groupmod -g "$DIR_GID" docker 2>/dev/null || true

        # Modify the docker user UID and assign to group
        usermod -u "$DIR_UID" -g "$DIR_GID" docker 2>/dev/null || true

        # Fix ownership of docker user's home directory
        chown -R docker:docker /home/docker 2>/dev/null || true
    fi
fi

# Install raddo package with editable install
gosu docker pip install -e . --quiet

# Execute the command as the docker user
exec gosu docker "$@"
