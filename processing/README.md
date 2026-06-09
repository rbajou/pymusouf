### Prerequisite
Install the package first; see [INSTALL](../INSTALL).

### Configuration and data sources
The processing pipeline supports two modes:
- **demo/sample mode** using lightweight inputs under [sample](../sample)
- **production mode** using local data roots configured in [config/config.yaml](../config/config.yaml) or a local override file

Available telescopes and runs are listed in [survey/survey.yaml](../survey/survey.yaml), while telescope geometry is defined in [telescope/telescopes.yaml](../telescope/telescopes.yaml).

The script [processing/tracks.py](tracks.py) reads raw files from the `raw` subfolder of the selected run directory and writes outputs to sibling folders such as `reco`, `log`, `png`, `json`, and `npy`.

---

## Step-by-step guide for a new user with `.dat` files

### Case 1 — the telescope is already present in the YAML catalogue

This is the simplest case.

1. **Check that the telescope already exists**
   - Open [telescope/telescopes.yaml](../telescope/telescopes.yaml) and confirm that your telescope code is listed.
   - Open [survey/survey.yaml](../survey/survey.yaml) and confirm that the run label you want to use exists for this telescope.

2. **Point the package to your local data root**
   - In a local Python/venv workflow, either edit [config/config.yaml](../config/config.yaml),
   - or better, copy [config/config.local.yaml.example](../config/config.local.yaml.example) to a local override file and set the `paths.data` value to the parent folder containing your telescope data.
   - In a Docker workflow, prefer [setup-docker.sh](../setup-docker.sh): it generates [docker-compose.yml](../docker-compose.yml) and sets `PYMUSOUF_DATA_DIR` and `PYMUSOUF_STRUCT_DIR` automatically, so you usually do not need to edit the config file for paths.

   Example logic:
   - if your files are stored under a folder like `.../SB/tomo/raw`, then `paths.data` or `PYMUSOUF_DATA_DIR` should point to the parent directory above `SB`.

3. **Disable sample mode for real data**
   - Set `runtime.use_sample_data: false` in the configuration.

4. **Place the raw files in the expected run folder**
   - The raw input files must be inside the `raw` subfolder of the run.
   - Accepted input names are files containing `dat` in their name, such as `.dat` or `.dat.gz`.

5. **Run a short test first**
   From the repository root or from the [processing](.) folder, run a small test with one file and a limited number of events.

6. **Launch the full tracking**
   Use a command such as:
   - telescope: the catalogue name, for example `SB` or `SNJ`
   - run: the run label defined in [survey/survey.yaml](../survey/survey.yaml), for example `tomo` or `calib`

7. **Check the outputs**
   The main output is the reconstructed dataframe `df_track.csv.gz` in the `reco` subfolder of the selected run.
   A log file is also written in the `log` subfolder.

### Case 2 — the telescope is not present in the catalogue

In this case, you first need to declare the instrument geometry.

1. **Create or import the channel mapping JSON**
   - Add the channel-to-bar mapping file under the telescope resources tree in [telescope](../telescope).
   - You can use [telescope/channel_to_bar.py](../telescope/channel_to_bar.py) as a starting point to generate a `mapping.json` file.

2. **Add the telescope definition**
   - Edit [telescope/telescopes.yaml](../telescope/telescopes.yaml).
   - Add:
     - telescope name
     - survey name
     - coordinates
     - azimuth and zenith
     - panel configurations
     - matrix type for each panel
     - mapping aliases

3. **Register the runs**
   - Edit [survey/survey.yaml](../survey/survey.yaml) and add the telescope under the chosen survey.
   - Define at least one run label such as `tomo` or `calib`.

4. **Prepare the local folder layout**
   - Put the user raw files in the `raw` subfolder of the run directory.
   - Make sure the directory tree matches the run path declared in [survey/survey.yaml](../survey/survey.yaml).

5. **Reload the package**
   - If you installed the package previously, reinstall it in editable mode or restart the Python session so the new YAML entries are taken into account.

6. **Run a small test first**
   Start with a single file and a small number of events to validate the mapping and geometry.

7. **Run the full reconstruction**
   Once the test output looks correct, process the complete dataset.

---

## Main command-line inputs

The main options of [processing/tracks.py](tracks.py) are:
- `--survey` or `-s`: survey name
- `--telescope` or `-t`: telescope name from the catalogue
- `--run` or `-r`: run label defined in [survey/survey.yaml](../survey/survey.yaml)
- `--nfiles` or `-nf`: maximum number of files to process
- `--nevents` or `-nev`: maximum number of events to reconstruct
- `--adc_calibration` or `-ac`: whether ADC calibration should be run before tracking
- `--residual_threshold`, `--min_samples`, `--max_trials`: RANSAC parameters

See also [cli/common_args.py](../cli/common_args.py) and [cli/processing_args.py](../cli/processing_args.py).

---

## Output

The reconstruction generates a compressed dataframe named `df_track.csv.gz` containing per-event tracking observables for each panel configuration.

During reconstruction, the script also creates a local `.args_cache.json` file to reuse argument defaults between runs.

---

### Event rate and images
See [processing/plot_images.py](plot_images.py).

This step produces reconstructed event-rate views and 2D angular histograms.

---

### Pipeline to opacity estimates
See [processing/muography.py](muography.py).

This stage stores combined observables and ray-length products in `muography.h5`.
