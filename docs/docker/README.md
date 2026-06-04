# Docker for pymusouf

This page is the main Docker reference for running pymusouf.

## Files involved

- Dockerfile: Python image and package installation
- docker-compose.yml: local container orchestration
- setup-docker.sh: interactive local/NAS configuration generator
- Makefile.docker: optional shortcuts

## Quick start

From the project root:

```bash
./setup-docker.sh local
docker compose build
docker compose up -d
```

Run a script:

```bash
docker compose exec pymusouf python processing/tracks.py
```

Open an interactive shell in the container:

```bash
docker compose exec pymusouf /bin/bash
```

If bash is unavailable in the image, use:

```bash
docker compose exec pymusouf /bin/sh
```

## Data configuration

setup-docker.sh supports two modes:

1. local: data/structure paths from the host machine
2. nas: path to a NAS share mounted on the host machine

The script generates docker-compose.yml with volumes and environment variables adapted to the selected mode.

Important: generated absolute paths in docker-compose.yml are machine-specific. Re-run setup-docker.sh on each machine instead of copying docker-compose.yml as-is.

## Clickable paths in VS Code

To get clickable host paths in VS Code logs, Docker config must expose host absolute paths in the container and set:

- PYMUSOUF_DATA_DIR
- PYMUSOUF_STRUCT_DIR
- PYMUSOUF_SAMPLE_DIR

The current project configuration is already aligned with this.

## Useful commands

```bash
# List containers
docker compose ps

# Stream logs
docker compose logs -f pymusouf

# Stop cleanly
docker compose down

# Full rebuild (if Dockerfile or dependencies changed)
docker compose up -d --build
```

## Is Makefile.docker required?

Makefile.docker is optional.

- Required: no
- Useful: yes, for shorter commands and fewer typing mistakes

Everything works without it by using docker compose and setup-docker.sh directly.

## Concepts

For a conceptual overview (image, container, volume, compose), see docs/docker/CONCEPTS.md.
