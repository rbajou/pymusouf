#!/bin/bash
# Docker setup script for pymusouf
# Usage: ./setup-docker.sh [local|nas]

set -e

MODE="${1:-local}"

echo "Docker setup for pymusouf"
echo "======================================="
echo ""

if [ "$MODE" = "local" ]; then
  echo "Mode: LOCAL DATA"
  echo ""
  read -p "Path to your data files (e.g. /Users/john/data): " DATA_PATH
  read -p "Path to your structure files (e.g. /Users/john/struct): " STRUCT_PATH

  if [ ! -d "$DATA_PATH" ]; then
    echo "Error: $DATA_PATH does not exist"
    exit 1
  fi

  if [ ! -d "$STRUCT_PATH" ]; then
    echo "Error: $STRUCT_PATH does not exist"
    exit 1
  fi

  # Create docker-compose.yml
  cat > docker-compose.yml << EOF
services:
  pymusouf:
    build: .
    container_name: pymusouf-dev
    volumes:
      - $DATA_PATH:/data
      - $DATA_PATH:$DATA_PATH
      - $STRUCT_PATH:/struct
      - $STRUCT_PATH:$STRUCT_PATH
      - ./output:/output
      - .:/pymusouf
    environment:
      - PYTHONUNBUFFERED=1
      - PYMUSOUF_DATA_DIR=$DATA_PATH
      - PYMUSOUF_STRUCT_DIR=$STRUCT_PATH
      - PYMUSOUF_SAMPLE_DIR=/pymusouf/sample
    stdin_open: true
    tty: true
EOF

  echo "docker-compose.yml configured for LOCAL mode"
  echo "   Data: $DATA_PATH -> /data"
  echo "   Structure: $STRUCT_PATH -> /struct"

elif [ "$MODE" = "nas" ]; then
  echo "Mode: NAS ACCESS (SMB/CIFS)"
  echo ""
  read -p "Path to mounted NAS root (default: /Volumes/NAS-sysvol): " NAS_PATH
  NAS_PATH="${NAS_PATH:-/Volumes/NAS-sysvol}"

  read -p "Data subfolder inside Equipe (default: data): " DATA_REL
  read -p "Structure subfolder inside Equipe (default: structure): " STRUCT_REL
  DATA_REL="${DATA_REL:-data}"
  STRUCT_REL="${STRUCT_REL:-structure}"

  EQUIPE_PATH="$NAS_PATH/Equipe"

  # Case 1: NAS already mounted on host (bind mounts)
  if [ -d "$EQUIPE_PATH" ]; then
    DATA_PATH="$EQUIPE_PATH/$DATA_REL"
    STRUCT_PATH="$EQUIPE_PATH/$STRUCT_REL"

    if [ ! -d "$DATA_PATH" ]; then
      echo "Error: data folder does not exist: $DATA_PATH"
      exit 1
    fi

    if [ ! -d "$STRUCT_PATH" ]; then
      echo "Error: structure folder does not exist: $STRUCT_PATH"
      exit 1
    fi

    # Create docker-compose.yml (host mount mode)
    cat > docker-compose.yml << EOF
services:
  pymusouf:
    build: .
    container_name: pymusouf-dev
    volumes:
      - $DATA_PATH:/data
      - $DATA_PATH:$DATA_PATH
      - $STRUCT_PATH:/struct
      - $STRUCT_PATH:$STRUCT_PATH
      - ./output:/output
      - .:/pymusouf
    environment:
      - PYTHONUNBUFFERED=1
      - PYMUSOUF_DATA_DIR=$DATA_PATH
      - PYMUSOUF_STRUCT_DIR=$STRUCT_PATH
      - PYMUSOUF_SAMPLE_DIR=/pymusouf/sample
    stdin_open: true
    tty: true
EOF

    echo "docker-compose.yml configured for NAS mode (host-mounted SMB)"
    echo "   NAS root: $NAS_PATH"
    echo "   Shared folder: $EQUIPE_PATH"
    echo "   Data: $DATA_PATH -> /data"
    echo "   Structure: $STRUCT_PATH -> /struct"

  # Case 2: NAS mount managed by Docker via CIFS volume
  else
    echo "Mounted NAS not found at: $EQUIPE_PATH"
    echo "Switching to Docker-managed CIFS mount mode."

    # Load existing .env if present
    if [ -f ".env" ]; then
      set -a
      # shellcheck disable=SC1091
      source .env
      set +a
    fi

    NAS_HOST="${NAS_HOST:-10.1.40.6}"
    NAS_SHARE="${NAS_SHARE:-Equipe}"
    NAS_VERS="${NAS_VERS:-3.0}"
    NAS_PASSWORD_VALUE="${NAS_PASSWORD:-${NAS_PASS:-}}"

    # Ask only for missing values
    if [ -z "$NAS_USER" ]; then
      read -p "NAS username: " NAS_USER
    fi

    if [ -z "$NAS_PASSWORD_VALUE" ]; then
      read -s -p "NAS password: " NAS_PASSWORD_VALUE
      echo ""
    fi

    if [ -z "$NAS_USER" ] || [ -z "$NAS_PASSWORD_VALUE" ]; then
      echo "Error: missing NAS credentials"
      exit 1
    fi

    # Persist credentials for reuse
    cat > .env << EOF
NAS_USER=$NAS_USER
NAS_PASSWORD=$NAS_PASSWORD_VALUE
NAS_HOST=$NAS_HOST
NAS_SHARE=$NAS_SHARE
NAS_VERS=$NAS_VERS
EOF
    chmod 600 .env

    # Create docker-compose.yml (Docker-managed CIFS mode)
    cat > docker-compose.yml << EOF
services:
  pymusouf:
    build: .
    container_name: pymusouf-dev
    volumes:
      - nas_equipe:/nas_equipe
      - ./output:/output
      - .:/pymusouf
    environment:
      - PYTHONUNBUFFERED=1
      - PYMUSOUF_DATA_DIR=/nas_equipe/$DATA_REL
      - PYMUSOUF_STRUCT_DIR=/nas_equipe/$STRUCT_REL
      - PYMUSOUF_SAMPLE_DIR=/pymusouf/sample
    stdin_open: true
    tty: true

volumes:
  nas_equipe:
    driver: local
    driver_opts:
      type: cifs
      o: "username=\${NAS_USER},password=\${NAS_PASSWORD},vers=\${NAS_VERS},rw"
      device: "//$NAS_HOST/$NAS_SHARE"
EOF

    echo "docker-compose.yml configured for NAS mode (Docker-managed CIFS)"
    echo "   SMB endpoint: //$NAS_HOST/$NAS_SHARE"
    echo "   Data in container: /nas_equipe/$DATA_REL"
    echo "   Structure in container: /nas_equipe/$STRUCT_REL"
    echo "   Credentials source: .env (saved for next runs)"
  fi

else
    echo "Invalid mode. Use: ./setup-docker.sh [local|nas]"
    exit 1
fi

echo ""
echo "Next steps:"
echo "1. docker compose build"
echo "2. docker compose up -d"
echo "3. docker compose exec pymusouf /bin/bash"
echo ""
echo "Read docs/docker/README.md for details"
