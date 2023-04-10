# MultiNeRF: Testing with Ref-NeRF

Copy of original repo from [Google Research](https://github.com/google-research/multinerf.git)

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
You'll probably also need to update your JAX installation to support GPUs or TPUs.

Install Jaxlib version that matches CUDA 11.3:
```
pip install jax==0.4.1
pip install https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.4.1+cuda11.cudnn86-cp310-cp310-manylinux2014_x86_64.whl
```
Monitor GPU usage as batch_size must be reduced to match available GPU memory:
```
pip install nvitop
```
Set LD library path to correct environment:
```
LD_LIBRARY_PATH=/home/martin/anaconda3/envs/points/lib/
export LD_LIBRARY_PATH
```
### OOM errors

As I use RTC2080TI with only 11GB memory, I have to reduce batch size, otherwise I 
get OOM error. 

If you do this, but want to preserve quality, be sure to increase the number
of training iterations and decrease the learning rate by whatever scale factor you
decrease batch size by.

## Using your own data

Download datasets from here:

Dataset 1: [mouse-2](https://1drv.ms/u/s!AtwBlzVMECHCpCPPnW6SjB4GmFBF?e=xZPY5M)  
Dateset 2: [kettle-2]()


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

See below for more detailed instructions on either using COLMAP to calculate poses or writing your own dataset loader (if you already have pose data from another source, like SLAM or RealityCapture).
