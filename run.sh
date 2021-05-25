#!/usr/bin/env bash

# #################### get env directories
USE_GPU=1
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
# DATASET ROOT
DATASET_ROOT=${CODE_ROOT}/dataset
if [ ! -d "${DATASET_ROOT}" ]; then
  echo "DATASET_ROOT= ${DATASET_ROOT}, not exists, exit"
  exit 1
fi
# OS RELEASE
OS_ID=$(awk -F= '$1=="ID" { print $2 ;}' /etc/os-release)

echo "CONDA_ROOT= ${CONDA_ROOT}"
echo "CONDA_NEW_ENV= ${CONDA_NEW_ENV}"
echo "JUPYTER_ROOT= ${JUPYTER_ROOT}"
echo "CODE_ROOT= ${CODE_ROOT}"
echo "DATASET_ROOT= ${DATASET_ROOT}"
echo "OS_ID= ${OS_ID}"

# #################### activate conda env and check lib versions
# solve run problem in Jupyter Notebook
# conda in shell propagation issue - https://stackoverflow.com/questions/52779016/conda-command-working-in-command-prompt-but-not-in-bash-script/52813960#52813960
CONDA_CONFIG_FILE="${CONDA_ROOT}/etc/profile.d/conda.sh"
if [ ! -f "${CONDA_CONFIG_FILE}" ]; then
  echo "CONDA_CONFIG_FILE= ${CONDA_CONFIG_FILE}, not exists, exit"
  exit 1
fi
# shellcheck disable=SC1090
source "${CONDA_CONFIG_FILE}"

# ###### activate conda env
# conda env by name
# conda activate ${CONDA_NEW_ENV}
# conda env by prefix
conda activate ${JUPYTER_ROOT}/envs/${CONDA_NEW_ENV}
conda info --envs

# check tf versions
python -c "from tensorflow.python.client import device_lib; print(device_lib.list_local_devices())"
python -c "import tensorflow as tf; print(tf.__version__)"
# check np versions
python -c "import numpy as np; print(np.__version__)"
# check torch versions
python -c "import torch; print(torch.__version__)"
python -c "import torchvision; print(torchvision.__version__)"

# #################### get 1st input argument as TYPE
TYPE=help
if [ -z "$1" ]; then
    echo "[Warning] TYPE is not set, using '${TYPE}' as default"
else
    TYPE=$(echo "$1" | tr '[:upper:]' '[:lower:]')
    echo "[Info] TYPE is ${TYPE}"
fi

# #################### execute according to TYPE
########## check
if [ "$TYPE" = "help" ]; then
  echo "Run for structuring: ./run.sh [TYPE] [Parameters]"
  echo "[TYPE] can be the following options:"
  echo "  ./run.sh help: help for ./run.sh"
  echo "  ./run.sh check: check conda environment"
  echo "  ./run.sh fix: fix conda environment when you restart from pausing"
  echo "  ./run.sh seg_extract: feature extraction for scene segmentation, including shot detection, feature extraction, and so on"
  echo "  ./run.sh seg_gt: generate gt files for scene segmentation"
  echo "  ./run.sh seg_train: train for scene segmentation"
  echo "  ./run.sh tag_extract: feature extraction for tagging"
  echo "  ./run.sh tag_gt: generate gt files for tagging"
  echo "  ./run.sh tag_train [CONFIG_FILE]: train for tagging"
  echo "            CONFIG_FILE: optional, config file path, default is ${CODE_ROOT}/MultiModal-Tagging/configs/config.structuring.5k.yaml"
  echo "  ./run.sh test [CHECKPOINT_DIR] [OUTPUT_FILE_PATH] [TEST_SCENE_VIDEOS_DIR]"
  echo "            CHECKPOINT_DIR: relative model dir under ${CODE_ROOT}/MultiModal-Tagging/, such as 'checkpoints/structuring_train5k/export/step_7000_0.7875'"
  echo "            OUTPUT_FILE_PATH: optional, relative output file path under ${CODE_ROOT}/MultiModal-Tagging/, default './results/structuring_tagging_5k.json'"
  echo "            TEST_SCENE_VIDEOS_DIR: optional, test scene videos directory, default is ${DATASET_ROOT}/structuring/structuring_dataset_test_5k/scene_video"
  echo "  ./run.sh eval [RESULT_FILE_PATH] [GT_FILE_PATH]"
  echo "            RESULT_FILE_PATH: result file for test data, such as './results/structuring_tagging_5k.json'"
  echo "            GT_FILE_PATH: gt file for test data, such as ${DATASET_ROOT}/gt_json/test100.json"

  exit 0
elif [ "$TYPE" = "check" ]; then
  echo "[Info] just check the conda env ${CONDA_NEW_ENV}"

  exit 0
elif [ "$TYPE" = "fix" ]; then
  echo "[Info] fix the environment when you restart from pausing"

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

  # #################### recreate ipython kernels
  # conda in shell propagation issue - https://stackoverflow.com/questions/52779016/conda-command-working-in-command-prompt-but-not-in-bash-script/52813960#52813960
  # shellcheck source=/opt/conda/etc/profile.d/conda.sh
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"

  # add envs_dirs
  conda config --add envs_dirs ${JUPYTER_ROOT}/envs
  conda config --show | grep env

  # ###### create env and activate
  # TensorFlow 1.14 GPU dependencies - https://www.tensorflow.org/install/source#gpu
  # create env by prefix
  conda activate ${JUPYTER_ROOT}/envs/${CONDA_NEW_ENV}
  # create env by name
  # conda activate ${CONDA_NEW_ENV}
  conda info --envs
  conda info

  # create a kernel for conda env
  python -m ipykernel install --user --name ${CONDA_NEW_ENV} --display-name "TAAC2021 (${CONDA_NEW_ENV})"

  exit 0
########## seg_extract - extracted features for scene segmentation
elif [ "$TYPE" = "seg_extract" ]; then
  # ########## shot detection: generating shot_keyf / shot_split_video / shot_stats / shot_txt, in structuring/structuring_dataset_train_5k
  cd ${CODE_ROOT}/SceneSeg/pre || exit 1
  pwd
  OUTPUT_SCENE_SEG_TRAIN_5K_DIR=${DATASET_ROOT}/structuring/structuring_dataset_train_5k
  if [ ! -d "${OUTPUT_SCENE_SEG_TRAIN_5K_DIR}" ]; then
    echo "[Info] structuring_dataset_train_5k not exists, generating: ${OUTPUT_SCENE_SEG_TRAIN_5K_DIR}"
    time python -u ShotDetect/shotdetect.py --video_dir ${DATASET_ROOT}/videos/train_5k_A/ \
                                         --save_dir ${OUTPUT_SCENE_SEG_TRAIN_5K_DIR}/
  else
    echo "[Info] structuring_dataset_train_5k exists, no need to generate: ${OUTPUT_SCENE_SEG_TRAIN_5K_DIR}"
  fi

  # ########## feature extraction: generating aud_feat / place_feat, in structuring/structuring_dataset_train_5k
  cd ${CODE_ROOT}/SceneSeg/pre || exit 1
  pwd
  # audio_feat
  OUTPUT_SCENE_SEG_TRAIN_AUDIO_FEAT_DIR=${OUTPUT_SCENE_SEG_TRAIN_5K_DIR}/aud_feat
  if [ ! -d ${OUTPUT_SCENE_SEG_TRAIN_AUDIO_FEAT_DIR} ]; then
    echo "[Info] audio features not exists, generating: OUTPUT_SCENE_SEG_TRAIN_AUDIO_FEAT_DIR= ${OUTPUT_SCENE_SEG_TRAIN_AUDIO_FEAT_DIR}"
    time python -u audio/extract_feat.py --data_root "${OUTPUT_SCENE_SEG_TRAIN_5K_DIR}"
  else
    echo "[Info] audio features exist, no need to generate: OUTPUT_SCENE_SEG_TRAIN_AUDIO_FEAT_DIR= ${OUTPUT_SCENE_SEG_TRAIN_AUDIO_FEAT_DIR}"
  fi
  # place_feat
  OUTPUT_SCENE_SEG_TRAIN_PLACE_FEAT_DIR=${OUTPUT_SCENE_SEG_TRAIN_5K_DIR}/place_feat
  if [ ! -d ${OUTPUT_SCENE_SEG_TRAIN_PLACE_FEAT_DIR} ]; then
    echo "[Info] place features not exists, generating: OUTPUT_SCENE_SEG_TRAIN_PLACE_FEAT_DIR= ${OUTPUT_SCENE_SEG_TRAIN_PLACE_FEAT_DIR}"
    time python -u place/extract_feat.py --data_root "${OUTPUT_SCENE_SEG_TRAIN_5K_DIR}"
  else
    echo "[Info] place features exist, no need to generate: OUTPUT_SCENE_SEG_TRAIN_PLACE_FEAT_DIR= ${OUTPUT_SCENE_SEG_TRAIN_PLACE_FEAT_DIR}"
  fi

  exit 0
########## seg_gt - generate gt files for scene segmentation training
elif [ "$TYPE" = "seg_gt" ]; then
  # ########## generating dataset: generating meta / labels, in structuring/structuring_dataset_train_5k
  cd ${CODE_ROOT}/SceneSeg/pre || exit 1
  pwd
  time python -u gt_generator.py --video_dir ${DATASET_ROOT}/videos/train_5k/ \
                              --data_root ${DATASET_ROOT}/structuring/structuring_dataset_train_5k/ \
                              --input_annotation ${DATASET_ROOT}/structuring/GroundTruth/train5k.txt

  exit 0
########## seg_train
elif [ "$TYPE" = "seg_train" ]; then
  cd ${CODE_ROOT}/SceneSeg/lgss || exit 1
  pwd

  # ########## seg_train
  echo "[Info] scene segmentation train"
  time python -u run.py config/train5k.py

  exit 0
########## tag_extract - extracted features are provided
elif [ "$TYPE" = "tag_extract" ]; then
  DATASET_ROOT=${CODE_ROOT}/dataset
  echo "[Info] extracted features are provided, no need to execute TYPE= ${TYPE}"
  echo "  video features:            ${DATASET_ROOT}/structuring/train5k_split_video_feats/video_npy/"
  echo "  audio features:            ${DATASET_ROOT}/structuring/train5k_split_video_feats/audio_npy/"
  echo "  video banner image files:  ${DATASET_ROOT}/structuring/train5k_split_video_feats/image_jpg/"
  echo "  text ocr and asr features: ${DATASET_ROOT}/structuring/train5k_split_video_feats/text_txt/"

  exit 0
########## tag_gt - generate gt files for tagging training
elif [ "$TYPE" = "tag_gt" ]; then
  cd ${CODE_ROOT}/MultiModal-Tagging || exit 1
  pwd

  # ########## 从原始视频（videos/train_5k_A）和标注信息（structuring/GourndTruth/train5k.txt, json格式）里生成标注文件（structuring/GroundTruth/structuring_tagging_info.txt, csv格式）
  echo "[Info] generating structuring/GroundTruth/structuring_tagging_info.txt for training data"
  TAGGING_INFO_FILE=${DATASET_ROOT}/tagging/GroundTruth/tagging_info.txt
  if [ ! -f ${TAGGING_INFO_FILE} ]; then
    echo "[Info] tagging_info.txt not exists, generating: ${TAGGING_INFO_FILE}"
    time python -u scripts/preprocess/json2info.py --video_dir ${DATASET_ROOT}/videos/train_5k_A \
                                                --json_path ${DATASET_ROOT}/structuring/GourndTruth/train5k.txt \
                                                --save_path ${DATASET_ROOT}/structuring/GroundTruth/structuring_tagging_info.txt \
                                                --convert_type structuring \
                                                --split_video_dir ${DATASET_ROOT}/structuring/train5k_split_video/
  else
    echo "[Info] tagging_info.txt exists, no need to generate: ${TAGGING_INFO_FILE}"
  fi

  # ########## 从标签字典文件、训练集标注文件（structuring/GroundTruth/structuring_tagging_info.txt, csv格式）、特征文件（structuring/train5k_split_video_feats/）等生成训练数据集
  echo "[Info] generating gt files for tagging training data"
  OUTPUT_TRAIN_FILE=${DATASET_ROOT}/tagging/GroundTruth/datafile/train.txt
  OUTPUT_VAL_FILE=${DATASET_ROOT}/tagging/GroundTruth/datafile/val.txt
  if [[ ! -f ${OUTPUT_TRAIN_FILE} ]] || [[ ! -f ${OUTPUT_VAL_FILE} ]]; then
    echo "[Info] gt files for tagging not exists, generating: ${OUTPUT_TRAIN_FILE}, ${OUTPUT_VAL_FILE}"
    time python -u scripts/preprocess/generate_datafile.py --info_file ${DATASET_ROOT}/structuring/GroundTruth/structuring_tagging_info.txt \
                                                        --out_file_dir ${DATASET_ROOT}/structuring/GroundTruth/datafile/ \
                                                        --tag_dict_path ${DATASET_ROOT}/label_id.txt \
                                                        --frame_npy_folder ${DATASET_ROOT}/structuring/train5k_split_video_feats/video_npy/ \
                                                        --audio_npy_folder ${DATASET_ROOT}/structuring/train5k_split_video_feats/audio_npy/ \
                                                        --text_txt_folder ${DATASET_ROOT}/structuring/train5k_split_video_feats/text_txt/ \
                                                        --image_folder ${DATASET_ROOT}/structuring/train5k_split_video_feats/image_jpg/
  else
    echo "[Info] gt files for tagging exists, no need to generate: ${OUTPUT_TRAIN_FILE}, ${OUTPUT_VAL_FILE}"
  fi

  exit 0
########## tag_train
elif [ "$TYPE" = "tag_train" ]; then
  cd ${CODE_ROOT}/MultiModal-Tagging || exit 1
  pwd

  # ########## CONFIG_FILE
  CONFIG_FILE=${CODE_ROOT}/MultiModal-Tagging/configs/config.structuring.5k.yaml
  if [ -z "$2" ]; then
      echo "[Warning] CONFIG_FILE is not set for TYPE= ${TYPE}, using default: ${CONFIG_FILE}"
  else
      CONFIG_FILE="$2"
      echo "[Info] CONFIG_FILE is ${CONFIG_FILE}"
  fi
  # check config file
  if [ ! -f "${CONFIG_FILE}" ]; then
    echo "[Error] config file not exists, CONFIG_FILE= ${CONFIG_FILE}"
    exit 1
  fi

  # ########## tagging train
  echo "[Info] tagging train with config= ${CONFIG_FILE}"
  python -u scripts/train_tagging.py --config "${CONFIG_FILE}"

  exit 0
########## test
elif [ "$TYPE" = "test" ]; then
  DATASET_ROOT=${CODE_ROOT}/dataset

  # ########## get checkpoints / tag_id_file / test_videos_dir / output_file from cmd arguments
  # CHECKPOINT_DIR as $2: must be set, such as "checkpoints/structuring_train5k/export/step_7000_0.7875"
  if [ -z "$2" ]; then
    echo "[Error] CHECKPOINT_DIR is not set, please set it when type= ${TYPE}"
    exit 1
  else
    CHECKPOINT_DIR="$2"
    # check
    if [ ! -d "${CODE_ROOT}/MultiModal-Tagging/${CHECKPOINT_DIR}" ]; then
      echo "[Error] checkpoint not exists, CHECKPOINT_DIR= ${CODE_ROOT}/MultiModal-Tagging/${CHECKPOINT_DIR}"
      exit 1
    fi
  fi
  # OUTPUT_FILE_PATH as $3: optional, output file path
  OUTPUT_FILE_PATH="${CODE_ROOT}/MultiModal-Tagging/results/structuring_tagging_5k.json"
  if [ -z "$3" ] ;then
      echo "[Warning] OUTPUT_FILE_PATH is not set, use default ${OUTPUT_FILE_PATH}"

      # create default result directory
      DEFAULT_OUTPUT_DIR="${CODE_ROOT}/MultiModal-Tagging/results"
      if [ ! -d "${DEFAULT_OUTPUT_DIR}" ]; then
        echo "[Warning] default results directory not exists, create it, DEFAULT_OUTPUT_DIR= ${DEFAULT_OUTPUT_DIR}"
        mkdir -p "${DEFAULT_OUTPUT_DIR}"
      fi
  else
    OUTPUT_FILE_PATH="$3"
  fi
  # TEST_SCENE_VIDEOS_DIR as $4: optional, scene videos directory for test
  TEST_SCENE_VIDEOS_DIR=${DATASET_ROOT}/structuring/test5k_split_video
  if [ -z "$4" ] ;then
      echo "[Warning] TEST_SCENE_VIDEOS_DIR is not set, use default ${TEST_SCENE_VIDEOS_DIR}"
  else
    TEST_SCENE_VIDEOS_DIR="$4"
  fi
  # TAG_ID_FILE
  TAG_ID_FILE=${DATASET_ROOT}/label_id.txt

  # ########## check arguments
  echo "[Info] test with parameters:"
  echo "  CHECKPOINT_DIR= ${CHECKPOINT_DIR}"
  echo "  TAG_ID_FILE= ${TAG_ID_FILE}"
  echo "  OUTPUT_FILE_PATH= ${OUTPUT_FILE_PATH}"
  echo "  TEST_SCENE_VIDEOS_DIR= ${TEST_SCENE_VIDEOS_DIR}"

  # #################### shot detection: generating shot_keyf / shot_split_video / shot_stats / shot_txt, in structuring/structuring_dataset_test_5k
  cd ${CODE_ROOT}/SceneSeg/pre || exit 1
  pwd
  OUTPUT_SCENE_SEG_TEST_5K_DIR=${DATASET_ROOT}/structuring/structuring_dataset_test_5k
  if [ ! -d "${OUTPUT_SCENE_SEG_TEST_5K_DIR}" ]; then
    echo "[Info] structuring_dataset_test_5k not exists, generating: ${OUTPUT_SCENE_SEG_TEST_5K_DIR}"
    time python -u ShotDetect/shotdetect.py --video_dir ${DATASET_ROOT}/videos/test_5k_A/ \
                                         --save_dir ${OUTPUT_SCENE_SEG_TEST_5K_DIR}/
  else
    echo "[Info] structuring_dataset_test_5k exists, no need to generate: ${OUTPUT_SCENE_SEG_TEST_5K_DIR}"
  fi

  # #################### feature extraction: generating aud_feat / place_feat, in structuring/structuring_dataset_test_5k
  cd ${CODE_ROOT}/SceneSeg/pre || exit 1
  pwd
  # audio_feat
  OUTPUT_SCENE_SEG_TEST_AUDIO_FEAT_DIR=${OUTPUT_SCENE_SEG_TEST_5K_DIR}/aud_feat
  if [ ! -d ${OUTPUT_SCENE_SEG_TEST_AUDIO_FEAT_DIR} ]; then
    echo "[Info] audio features not exists, generating: OUTPUT_SCENE_SEG_TEST_AUDIO_FEAT_DIR= ${OUTPUT_SCENE_SEG_TEST_AUDIO_FEAT_DIR}"
    time python -u audio/extract_feat.py --data_root "${OUTPUT_SCENE_SEG_TEST_5K_DIR}"
  else
    echo "[Info] audio features exist, no need to generate: OUTPUT_SCENE_SEG_TEST_AUDIO_FEAT_DIR= ${OUTPUT_SCENE_SEG_TEST_AUDIO_FEAT_DIR}"
  fi
  # place_feat
  OUTPUT_SCENE_SEG_TEST_PLACE_FEAT_DIR=${OUTPUT_SCENE_SEG_TEST_5K_DIR}/place_feat
  if [ ! -d ${OUTPUT_SCENE_SEG_TEST_PLACE_FEAT_DIR} ]; then
    echo "[Info] place features not exists, generating: OUTPUT_SCENE_SEG_TEST_PLACE_FEAT_DIR= ${OUTPUT_SCENE_SEG_TEST_PLACE_FEAT_DIR}"
    time python -u place/extract_feat.py --data_root "${OUTPUT_SCENE_SEG_TEST_5K_DIR}"
  else
    echo "[Info] place features exist, no need to generate: OUTPUT_SCENE_SEG_TEST_PLACE_FEAT_DIR= ${OUTPUT_SCENE_SEG_TEST_PLACE_FEAT_DIR}"
  fi

  # ########## scene segmentation
  cd ${CODE_ROOT}/SceneSeg/lgss || exit 1
  pwd
  echo "[Info] begin to segment scene, using model '../run/sceneseg_train5k/model_best.pth.tar'"
  #time python -u run_inference.py --config config/common_test.py \
                               #--data_root "${OUTPUT_SCENE_SEG_TEST_5K_DIR}" \
                               #--model_path "/home/tione/notebook/VideoStructuring/SceneSeg/run/sceneseg_train5k/model_best.pth.tar" \
                               #--video_dir "/home/tione/notebook/VideoStructuring/dataset/videos/test_5k_A" \
                               #--output_root "${TEST_SCENE_VIDEOS_DIR}"

  # ########## tagging
  cd ${CODE_ROOT}/MultiModal-Tagging || exit 1
  pwd
  echo "[Info] begin to tag"
  if [ ! -d "${TEST_SCENE_VIDEOS_DIR}" ]; then
    echo "[Error] TEST_SCENE_VIDEOS_DIR not exists, TEST_SCENE_VIDEOS_DIR= ${TEST_SCENE_VIDEOS_DIR}"
    exit 1
  fi
  python -u scripts/inference_for_structuring.py --model_pb "${CHECKPOINT_DIR}" \
                                              --tag_id_file "${TAG_ID_FILE}" \
                                              --test_dir "${TEST_SCENE_VIDEOS_DIR}" \
                                              --output_json "${OUTPUT_FILE_PATH}" \
                                              --use_gpu ${USE_GPU}

  exit 0/
########## evaluate
elif [ "$TYPE" = "evaluate" ]; then
  cd ${CODE_ROOT}/MultiModal-Tagging || exit 1
  pwd

  # ########## get result_file / gt_file from cmd arguments
  # RESULT_FILE_PATH as $2: must be set, such as "./results/structuring_tagging_5k.json"
  if [ -z "$2" ]; then
    echo "[Error] RESULT_FILE_PATH is not set, please set it when type= ${TYPE}, such as './results/structuring_tagging_5k.json'"
    exit 1
  else
    RESULT_FILE_PATH="$2"
  fi
  # check
  if [ ! -f "${RESULT_FILE_PATH}" ]; then
    echo "[Error] RESULT_FILE_PATH not exists, RESULT_FILE_PATH= ${RESULT_FILE_PATH}"
    exit 1
  fi
  # GT_FILE_PATH as $3: must be set, such as "${DATASET_ROOT}/gt_json/test100.json"
  if [ -z "$3" ]; then
    echo "[Error] GT_FILE_PATH is not set, please set it when type= ${TYPE}"
    exit 1
  else
    GT_FILE_PATH="$3"
  fi
  # check
  if [ ! -f "${GT_FILE_PATH}" ]; then
    echo "[Error] GT_FILE_PATH not exists, GT_FILE_PATH= ${GT_FILE_PATH}, , such as ${DATASET_ROOT}/gt_json/test100.json"
    exit 1
  fi

  # ########## evaluate
  echo "[Info] evaluate with parameters:"
  echo "  RESULT_FILE_PATH= ${RESULT_FILE_PATH}"
  echo "  GT_FILE_PATH= ${GT_FILE_PATH}"
  time python -u scripts/structuring_evaluation.py --pred "${RESULT_FILE_PATH}" \
                                                --gt "${GT_FILE_PATH}"
  python -u scripts/structuring_evaluation.py --gt ../dataset/gt_json/test100.json --pred results/structuring_test5k_pred.json


  exit 0
else
  echo "[Error] type= $TYPE not supported"

  exit 0
fi
