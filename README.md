# DrTrack

Data Science Lab 2023 at ETH Zurich. Challenge: Tracking Droplets in Microfluidic Devices.

## Team
| Name                 | Email               | Github                                        |
| -------------------- | ------------------- | --------------------------------------------- |
| Sven Gutjahr | sgutjahr@ethz.ch | [svenlg](https://github.com/svenlg) |
| Tiago Hungerland     | thungerland@ethz.ch     | [thungerland](https://github.com/thungerland)         |
| Richard Danis        | rdanis@ethz.ch   | [richdanis](https://github.com/richdanis)     |
| Weronika Ormaniec        | weronika.ormaniec@gmail.com   | [werkaaa](https://github.com/werkaaa)     |
| Michael Vollenweider        | michavol@ethz.ch   | [michavol](https://github.com/michavol)     |

## Setup for Linux
- First create a python environment, using the requirements.txt
```python -m venv env_name```

- Activate the environment by running (in the directory of env_name):
```source env_name/bin/activate```

- Then install the requirements.txt (in the directory of requirements.txt):
```pip install -r requirements.txt```

## Download of Necessary Data
- To run the track_droplets.py, make sure to have saved an .nd2 image in data/01_raw.
Check out the config_track_droplets_.yaml in src/conf for further settings.

- To run evaluate_tracking.py, make sure to have installed the paired patch data (link to download). 
Store it inside the evaluation/03_features folder. Check out the config_evaluation.yaml in src/conf for further settings.

## Run Main Pipeline
- To run the main pipeline, configure your experiment in src/conf/config_track_droplets_.yaml and run:
```python src/track_droplets.py```
You can run only the setup part of the python script to setup the directory structure automatically.
- To run the evaluation pipeline, configure your experiment in src/conf/config_evaluation.yaml and run:
```python src/evaluate_droplets.py```

## Run on Euler
First you need to create your environment on euler.
```
$ module load gcc/8.2.0 python_gpu/3.11.2
$ python -m venv --system-site-packages env_name
$ source env_name/bin/activate
$ pip install -r euler/requirements.txt
```
Then you can need to submit the job with the following command:
(You may want to change the device and job duration)
```
$ sbatch euler/track_droplets.sh "env_name"
```