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

## Training on Euler

### Data

Move train.npy, validation.npy and test.npy into /cluster/scratch/$USER/data.
These contain the data in the following shape (subject to change):
    
```
train.npy: (N, 2, 2, 40, 40)
N: number of images
Second Dimension: 0: droplet, 1: label (corresponding droplet in the next frame)
Third Dimension: 0: BF channel, 1: DAPI channel 
```
There are around 24k samples in train.npy and around 2k in validation.npy and test.npy. (80/10/10 split)


### Download Model

Before you can run the training on euler you need to download the model locally and then upload it to euler.
Unfortunately it cannot be downloaded directly on euler.
For downloading you need to execute the following command locally:

```
python3 src/training/models/efficientnet.py
```
And then upload the model to euler:

```
$ scp src/training/efficientnet_b1.pth $USER@euler.ethz.ch:/cluster/home/$USER/DrTrack/src/training/models
```

### Virtual Environment
Then create the virtual environment on euler:

```
$ module load gcc/8.2.0 python_gpu/3.11.2

$ python -m venv --system-site-packages dslab_env

$ pip install requirements.txt
```

### Submit Job

Finally submit the job with (you may want to change the arguments in train.sh):
    
```
$ sbatch src/training/train.sh
```