#!/usr/bin/env bash

# #################### get env directories
# CONDA_ROOT
CONDA_CONFIG_ROOT_PREFIX=$(conda config --show root_prefix)
echo "CONDA_CONFIG_ROOT_PREFIX= ${CONDA_CONFIG_ROOT_PREFIX}"
get_conda_root_prefix() {
  TMP_POS=$(awk -v a="${CONDA_CONFIG_ROOT_PREFIX}" -v b="/" 'BEGIN{print index(a, b)}')
  TMP_POS=$((TMP_POS-1))
  if [ $TMP_POS -ge 0 ]; then
    echo "${CONDA_CONFIG_ROOT_PREFIX:${TMP_POS}}"
  else
    echo ""
  fi
}
CONDA_ROOT=$(get_conda_root_prefix)
if [ ! -d "${CONDA_ROOT}" ]; then
  echo "CONDA_ROOT= ${CONDA_ROOT}, not exists, exit"
  exit 1
fi
# CONDA ENV
CONDA_NEW_ENV=taac2021-structuring
# JUPYTER_ROOT
JUPYTER_ROOT=/home/tione/notebook
if [ ! -d "${JUPYTER_ROOT}" ]; then
  echo "JUPYTER_ROOT= ${JUPYTER_ROOT}, not exists, exit"
  exit 1
fi
# CODE ROOT
CODE_ROOT=${JUPYTER_ROOT}/VideoStructuring
if [ ! -d "${CODE_ROOT}" ]; then
  echo "CODE_ROOT= ${CODE_ROOT}, not exists, exit"
  exit 1
fi
# OS RELEASE
OS_ID=$(awk -F= '$1=="ID" { print $2 ;}' /etc/os-release)
OS_ID=${OS_ID//"\""/""}

echo "CONDA_ROOT= ${CONDA_ROOT}"
echo "CONDA_NEW_ENV= ${CONDA_NEW_ENV}"
echo "JUPYTER_ROOT= ${JUPYTER_ROOT}"
echo "CODE_ROOT= ${CODE_ROOT}"
echo "OS_ID= ${OS_ID}"

# #################### obviously set $1 to be 'run' to run ./init.sh
if [ -z "$1" ]; then
  ACTION="check"
else
  ACTION=$(echo "$1" | tr '[:upper:]' '[:lower:]')
fi
if [ "${ACTION}" != "run" ]; then
  echo "[Info] you don't set the ACTION as 'run', so just check the environment"
  exit 0
fi

# #################### install system libraries
if [ "${OS_ID}" == "ubuntu" ]; then
  echo "[Info] installing system libraries in ${OS_ID}"
  sudo apt-get update
  sudo apt-get install -y apt-utils
  sudo apt-get install -y libsndfile1-dev ffmpeg
elif [ "${OS_ID}" == "centos" ]; then
  echo "[Info] installing system libraries in ${OS_ID}"
  yum install -y libsndfile libsndfile-devel ffmpeg ffmpeg-devel
else
  echo "[Warning] os not supported for ${OS_ID}"
  exit 1
fi

# #################### use tsinghua conda sources
conda config --show channels
# conda config --remove channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
# conda config --remove channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
# conda config --remove channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
# conda config --show channels
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
# conda config --set show_channel_urls yes
# conda config --show channels

# #################### create conda env and activate
# conda in shell propagation issue - https://stackoverflow.com/questions/52779016/conda-command-working-in-command-prompt-but-not-in-bash-script/52813960#52813960
# shellcheck source=/opt/conda/etc/profile.d/conda.sh
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

# ###### create env and activate
# TensorFlow 1.14 GPU dependencies - https://www.tensorflow.org/install/source#gpu
# create env by prefix
conda create --prefix ${JUPYTER_ROOT}/envs/${CONDA_NEW_ENV} -y cudatoolkit=10.1 cudnn=7.6.0 python=3.7 ipykernel
conda activate ${JUPYTER_ROOT}/envs/${CONDA_NEW_ENV}
# create env by name
# conda create -n ${CONDA_NEW_ENV} -y cudatoolkit=10.0 cudnn=7.6.0 python=3.7 ipykernel
# conda activate ${CONDA_NEW_ENV}
conda info --envs

# #################### create jupyter kernel
# create a kernel for conda env
python -m ipykernel install --user --name ${CONDA_NEW_ENV} --display-name "TAAC2021 (${CONDA_NEW_ENV})"

# #################### install python libraries
# install related libraries
cd ${CODE_ROOT}/MultiModal-Tagging || exit
pwd

# tensorflow 1.14
#pip config set global.index-url https://mirrors.tencent.com/pypi/simple/
pip install -r requirement.txt
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install tensorflow-gpu==1.14 opencv-python librosa==0.7.2 tensorboardX==2.1 easyocr SpeechRecognition PocketSphinx
pip install mmcv -i http://mirrors.tencentyun.com/pypi/simple --trusted-host mirrors.tencentyun.com
pip install numba==0.48.0

# tensorflow 1.15 - solve problem - https://stackoverflow.com/questions/58479556/notimplementederror-cannot-convert-a-symbolic-tensor-2nd-target0-to-a-numpy
# pip config set global.index-url https://mirrors.tencent.com/pypi/simple/
# pip install -r requirement.txt
# pip install tensorflow-gpu==1.15 opencv-python torch==1.2.0 torchvision==0.4.0 librosa==0.7.2 tensorboardX==2.1
# pip install numpy==1.19.5
# pip install mmcv==1.0.3 -i http://mirrors.tencentyun.com/pypi/simple --trusted-host mirrors.tencentyun.com
# pip install numba==0.48.0

# check tensorflow GPU
python -c "import tensorflow as tf; tf.test.gpu_device_name()"
python -c "from tensorflow.python.client import device_lib; print(device_lib.list_local_devices())"

# check library versions
echo "[TensorFlow]"
python -c "import tensorflow as tf; print(tf.__version__)"
echo "[NumPy]"
python -c "import numpy as np; print(np.__version__)"
echo "[Torch]"
python -c "import torch; print(torch.__version__)"
python -c "import torchvision; print(torchvision.__version__)"
echo "[OpenCV]"
python -c "import cv2; print(cv2.__version__)"
echo "[librosa]"
python -c "import librosa; print(librosa.__version__)"
echo "[numba]"
python -c "import numba; print(numba.__version__)"
echo "[mmcv]"
python -c "import mmcv; print(mmcv.__version__)"
echo "[TensorBoardX]"
python -c "import tensorboardX; print(tensorboardX.__version__)"
