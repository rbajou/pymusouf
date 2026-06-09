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
docker compose exec pymusouf bash
# inside the container
py processing/tracks.py  # `py` is an alias to `python` in this Docker image
```

To quit the interactive shell session cleanly:

```bash
exit  # or press Ctrl+D
```

This exits the shell only; the container keeps running.

To stop the container cleanly from the project root:

```bash
docker compose down
```

As a non-interactive alternative, you can run one command directly:

```bash
docker compose exec pymusouf py processing/tracks.py  # `py` alias in the image (same as `python`)
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

# Rebuild when Dockerfile or dependencies changed
docker compose build
docker compose up -d
```

## Is Makefile.docker required?

Makefile.docker is optional.

- Required: no
- Useful: yes, for shorter commands and fewer typing mistakes

Everything works without it by using docker compose and setup-docker.sh directly.

## Concepts

For a conceptual overview (image, container, volume, compose), see docs/docker/CONCEPTS.md.
