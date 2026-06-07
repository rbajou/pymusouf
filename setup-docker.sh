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
version: '3.9'

services:
  pymusouf:
    image: ${PYMUSOUF_IMAGE:-ghcr.io/rbajou/pymusouf:latest}
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
    echo "Mode: NAS ACCESS"
    echo ""
    read -p "Path to mounted NAS (e.g. /Volumes/nas_backup): " NAS_PATH

  if [ ! -d "$NAS_PATH" ]; then
      echo "Error: $NAS_PATH does not exist or is not mounted"
      echo "   Mount the NAS on your host first"
    exit 1
  fi

    # Create docker-compose.yml
    cat > docker-compose.yml << EOF
version: '3.9'

services:
  pymusouf:
    image: ${PYMUSOUF_IMAGE:-ghcr.io/rbajou/pymusouf:latest}
    build: .
    container_name: pymusouf-dev
    volumes:
      - $NAS_PATH:/data
      - $NAS_PATH:$NAS_PATH
      - ./output:/output
      - .:/pymusouf
    environment:
      - PYTHONUNBUFFERED=1
      - PYMUSOUF_DATA_DIR=$NAS_PATH
      - PYMUSOUF_SAMPLE_DIR=/pymusouf/sample
    stdin_open: true
    tty: true
EOF

  echo "docker-compose.yml configured for NAS mode"
  echo "   Mounted NAS: $NAS_PATH -> /data"

else
    echo "Invalid mode. Use: ./setup-docker.sh [local|nas]"
    exit 1
fi

echo ""
  echo "Next steps:"
echo "1. docker compose pull"
echo "2. docker compose up -d --no-build"
echo "3. docker compose exec pymusouf /bin/bash"
echo ""
  echo "Read docs/docker/README.md for details"
