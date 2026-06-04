# Docker and pymusouf: key concepts

## Why Docker here

Docker provides a reproducible runtime for pymusouf:

- controlled Python version
- identical dependency installation
- fewer machine-to-machine differences

## Quick vocabulary

- Image: recipe built from the Dockerfile
- Container: running instance of an image
- Volume: shared files between host and container
- Docker Compose: service declaration and lifecycle commands

## Typical workflow

1. build: create the image
2. up: start the service
3. exec: run commands inside the container
4. down: stop the service

## Relation to pymusouf code changes

Source code is mounted into the container through a volume. Python module changes are therefore visible without reinstalling the package after each edit.

You usually rebuild when:

- Dockerfile changes
- Python/system dependencies change
- image structure needs regeneration

## Virtualenv vs Docker

- virtualenv: local Python-only isolation
- Docker: full runtime isolation

Both approaches are valid. Docker is often simpler when sharing a consistent environment across users.
