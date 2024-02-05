## Python Package Manager
- Anaconda: https://www.anaconda.com/
- Miniconda: https://docs.conda.io/projects/miniconda/en/latest/ (minimal version of Anaconda)

## Clone Repository
- GitHub Desktop for Windows: https://desktop.github.com/
- Or just via git: https://git-scm.com/download/win

## Install Dependencies
Run the following from the root directory of the DrTrack repository.
```shell
conda env create --name dr_track_test --file=environment_windows.yml
```

## Add nd2 Data
Add .nd2 file to data/01_raw directory

## Installs (in case environment_windows.yml does not work)
```shell
conda install conda-forge::omegaconf
pip install hydra-core --upgrade
pip install ott-jax
pip install nd2
pip install opencv-python
pip install -U scikit-image
pip install pandas
pip3 install torch torchvision torchaudio
pip install -U scikit-learn==1.2.2
python -m pip install -U matplotlib
pip install seaborn
pip install pyarrow
pip install wandb
```