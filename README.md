# MultiNeRF: Testing with Ref-NeRF

Copy of original repo from [Google Research](https://github.com/google-research/multinerf.git) leaving only essentials to train on your own data.

## Setup

```
# Clone the repo.
git clone https://github.com/google-research/multinerf.git
cd multinerf

# Make a conda environment.
conda create --name multinerf python=3.9
conda activate multinerf

# Prepare pip.
conda install pip
pip install --upgrade pip

# Install requirements.
pip install -r requirements.txt

# Manually install rmbrualla's `pycolmap` (don't use pip's! It's different).
git clone https://github.com/rmbrualla/pycolmap.git ./internal/pycolmap

# Confirm that all the unit tests pass.
./scripts/run_all_unit_tests.sh
```

### Install CUDA and cuDNN

Install Jaxlib version that matches CUDA 11.3 toolkit. First install CUDA and cuDNN in Anaconda using the following commands:
```
conda install -c "nvidia/label/cuda-11.3.1" cuda-toolkit
conda install -c "nvidia/label/cuda-11.3.1" cuda-nvcc
```

Check cuDNN version:
```
conda list cudnn
```

Check CUDA version:
```
nvcc -V
```
Now install jaxlib with same cuDNN version. Unfortunately, there is no pre-compiled version with cuDNN version 8.4. Could upgrade to 8.6, but conda-forge has no 8.6. Means need to downgrade both to 8.2.

```
pip install jax==0.4.1
pip install https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.4.1+cuda11.cudnn82-cp310-cp310-manylinux2014_x86_64.whl
```
Change cuDNN installation version in conda:
```
conda install -c anaconda cudnn==8.2.1

```

Set LD library path to correct environment:
```
LD_LIBRARY_PATH=/home/martin/anaconda3/envs/points/lib/
export LD_LIBRARY_PATH
```

### Check CUDA and cuDNN Installation

Check if jax detects GPU:

```
import jax
jax.default_backend()
jax.devices()
```

Check jaxlib cuDNN compile version:

```
pip show jaxlib
```

Last number indicates compile version like `0.4.1+cuda11.cudnn82` for version 8.2.


### OOM errors

As I use RTC2080TI with only 11GB memory, I have to reduce batch size, otherwise I 
get OOM error. 

If you do this, but want to preserve quality, be sure to increase the number
of training iterations and decrease the learning rate by whatever scale factor you
decrease batch size by.


Monitor GPU usage as batch_size must be reduced to match available GPU memory:
```
pip install nvitop
```

## Using your own data

Download datasets from here:

Dataset 1: [mouse-2](https://1drv.ms/u/s!AtwBlzVMECHCpCPPnW6SjB4GmFBF?e=xZPY5M)  
Dateset 2: [kettle-2](https://1drv.ms/u/s!AtwBlzVMECHCpCRmkp6JwkpzkFZA?e=NtLuyu)


Summary: first, calculate poses. Second, train MultiNeRF. Third, render a result video from the trained NeRF model.

1. Calculating poses (using COLMAP), requires a images/ folder:
```
DATA_DIR=/mnt/data/github/nerf/data/mine/kettle-2
bash scripts/local_colmap_and_resize.sh ${DATA_DIR}
```
2. Training MultiNeRF:
```
python -m train \
  --gin_configs=configs/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.checkpoint_dir = '${DATA_DIR}/checkpoints'" \
  --logtostderr
```
3. Rendering MultiNeRF:
```
python -m render \
  --gin_configs=configs/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.checkpoint_dir = '${DATA_DIR}/checkpoints'" \
  --gin_bindings="Config.render_dir = '${DATA_DIR}/render'" \
  --gin_bindings="Config.render_path = True" \
  --gin_bindings="Config.render_path_frames = 480" \
  --gin_bindings="Config.render_video_fps = 60" \
  --logtostderr
```
Your output video should now exist in the directory `my_dataset_dir/render/`.
