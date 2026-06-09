### Context
The repository contains Python packages to reconstruct and analyze muography data recorded during two surveys:
- La Soufrière de Guadeloupe, in the Lesser Antilles.
- Copahue volcano, on the Argentina–Chile border.

The detectors used in this study are scintillator-based hodoscopes developed at IP2I Lyon.

### Setup
Two setup modes are supported:

1. Docker (recommended)
2. Python virtual environment

Quick start (Docker-first):

```bash
./setup-docker.sh local
docker compose build                    # Build the Docker image from Dockerfile
docker compose up -d                    # Start the container in background
docker compose exec pymusouf bash  # Open an interactive shell in the running container
# inside the container
py processing/tracks.py  # `py` is an alias to `python` in this Docker image
```

For complete setup instructions (Docker and virtualenv), see [INSTALL](INSTALL).

For Docker details and troubleshooting, see [docs/docker/README.md](docs/docker/README.md).

### Repository layout
- package configuration in [config/config.yaml](config/config.yaml)
- telescope catalogue in [telescope/telescopes.yaml](telescope/telescopes.yaml)
- internal JSON channel-to-bar mappings in [telescope](telescope)
- lightweight demonstration data files in [sample](sample)

### Reconstruction
See [processing/README.md](processing/README.md).

### 3D modeling and inversion
See [inversion](inversion).