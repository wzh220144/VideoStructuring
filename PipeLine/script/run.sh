#./gen_seg_sample.sh
#./seg_train.sh
#sh -x /home/tione/notebook/VideoStructuring/PipeLine/script/seg_inference.sh
#rm -rf /home/tione/notebook/VideoStructuring/dataset/feats/
sh -x /home/tione/notebook/VideoStructuring/PipeLine/script/split_video.sh
python -u /home/tione/notebook/VideoStructuring/MultiModal-Tagging/scripts/inference_for_structuring.py
