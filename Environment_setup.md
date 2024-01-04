1. For GPU support make sure to do the following
2. Install the NVidia cuda-toolbox for cuda 11.8 on the machine
3. Create a conda environment with 3.11.7
4. ```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```
5. ```pip install ott-jax```
6. ```pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html```

