#python -u /home/tione/notebook/VideoStructuring/SceneSeg/pre/place/extract_feat.py --data_root=/home/tione/notebook/dataset/train_5k_A/shot_transnet_v2 --use_gpu=1
#python -u /home/tione/notebook/VideoStructuring/SceneSeg/pre/audio/extract_feat.py --data_root=/home/tione/notebook/dataset/train_5k_A/shot_transnet_v2
python -u /home/tione/notebook/VideoStructuring/SceneSeg/pre/place/extract_feat.py --data_root=/home/tione/notebook/dataset/test_5k_2nd/shot_transnet_v2 --use_gpu=1
python -u /home/tione/notebook/VideoStructuring/SceneSeg/pre/audio/extract_feat.py --data_root=/home/tione/notebook/dataset/test_5k_2nd/shot_transnet_v2
