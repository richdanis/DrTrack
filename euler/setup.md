# 1. Add to .bashrc:
```shell
env2lmod
module load gcc/8.2.0 python_gpu/3.11.2 eth_proxy
#export JAX_PLATFORMS=cuda # If gpu required
```

# 2. Create virtual environment
```shell
python -m venv --system-site-packages dr_track
source dr_track/bin/activate # You can also add this to the .bashrc
pip install --upgrade pip
pip install -r DrTrack/euler/requirements.txt
```

# 3. To setup remote branch
```shell
git fetch origin
git checkout -b fix-failing-tests origin/fix-failing-tests
```