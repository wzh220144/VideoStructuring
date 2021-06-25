from __future__ import print_function
from utilis import mkdir_ifmiss
from utilis.package import *
import random

from shotdetect.video_manager import VideoManager
from shotdetect.shot_manager import ShotManager

# For content-aware shot detection:
from shotdetect.video_splitter import is_ffmpeg_available,split_video_ffmpeg
from shotdetect.keyf_img_saver import generate_images,generate_images_txt

import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import glob
import tensorflow as tf
import numpy as np
import os
import ffmpeg
import cv2

class TransNetV2:
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), "transnetv2-weights/")
            if not os.path.isdir(model_dir):
                raise FileNotFoundError(f"[TransNetV2] ERROR: {model_dir} is not a directory.")
            else:
                print(f"[TransNetV2] Using weights from {model_dir}.")

        self._input_size = (27, 48, 3)
        try:
            self._model = tf.saved_model.load(model_dir)
        except OSError as exc:
            raise IOError(f"[TransNetV2] It seems that files in {model_dir} are corrupted or missing. "
                          f"Re-download them manually and retry. For more info, see: "
                          f"https://github.com/soCzech/TransNetV2/issues/1#issuecomment-647357796") from exc

    def predict_raw(self, frames: np.ndarray):
        assert len(frames.shape) == 5 and frames.shape[2:] == self._input_size, \
            "[TransNetV2] Input shape must be [batch, frames, height, width, 3]."
        frames = tf.cast(frames, tf.float32)

        logits, dict_ = self._model(frames)
        single_frame_pred = tf.sigmoid(logits)
        all_frames_pred = tf.sigmoid(dict_["many_hot"])

        return single_frame_pred, all_frames_pred

    def predict_frames(self, frames: np.ndarray):
        assert len(frames.shape) == 4 and frames.shape[1:] == self._input_size, \
            "[TransNetV2] Input shape must be [frames, height, width, 3]."

        def input_iterator():
            # return windows of size 100 where the first/last 25 frames are from the previous/next batch
            # the first and last window must be padded by copies of the first and last frame of the video
            no_padded_frames_start = 25
            no_padded_frames_end = 25 + 50 - (len(frames) % 50 if len(frames) % 50 != 0 else 50)  # 25 - 74

            start_frame = np.expand_dims(frames[0], 0)
            end_frame = np.expand_dims(frames[-1], 0)
            padded_inputs = np.concatenate(
                [start_frame] * no_padded_frames_start + [frames] + [end_frame] * no_padded_frames_end, 0
            )

            ptr = 0
            while ptr + 100 <= len(padded_inputs):
                out = padded_inputs[ptr:ptr + 100]
                ptr += 50
                yield out[np.newaxis]

        predictions = []

        for inp in input_iterator():
            single_frame_pred, all_frames_pred = self.predict_raw(inp)
            predictions.append((single_frame_pred.numpy()[0, 25:75, 0],
                                all_frames_pred.numpy()[0, 25:75, 0]))

            print("\r[TransNetV2] Processing video frames {}/{}".format(
                min(len(predictions) * 50, len(frames)), len(frames)
            ), end="")
        print("")

        single_frame_pred = np.concatenate([single_ for single_, all_ in predictions])
        all_frames_pred = np.concatenate([all_ for single_, all_ in predictions])

        return single_frame_pred[:len(frames)], all_frames_pred[:len(frames)]  # remove extra padded frames

    def predict_video(self, video_file, threshold = 0.5):
        print("[TransNetV2] Extracting frames from {}".format(video_file))
        video_stream, err = ffmpeg.input(video_file).output(
            "pipe:", format="rawvideo", pix_fmt="rgb24", s="48x27"
        ).run(capture_stdout=True, capture_stderr=True)

        video = np.frombuffer(video_stream, np.uint8).reshape([-1, 27, 48, 3])
        single_frame_predictions, all_frame_predictions = self.predict_frames(video)
        return [(i, x >= threshold) for i, x in enumerate(single_frame_predictions)]

def get_shots_from_cuts(cut_list, base_timecode, num_frames, start_frame = 0):
    # shot list, where shots are tuples of (Start FrameTimecode, End FrameTimecode).
    shot_list = []
    if not cut_list:
        shot_list.append((base_timecode + start_frame, base_timecode + num_frames))
        return shot_list
    # Initialize last_cut to the first frame we processed,as it will be
    # the start timecode for the first shot in the list.
    last_cut = base_timecode + start_frame
    t = 0
    for cut in cut_list:
        t = cut
        cut = base_timecode + cut
        shot_list.append((last_cut, cut))
        last_cut = cut
    if t >= num_frames - 1:
        shot_list.append((last_cut, base_timecode + num_frames))

    # Last shot is from last cut to end of video.
    shot_list.append((last_cut, base_timecode + num_frames))
    return shot_list

def main(device, model, args, video_path, data_root):
    cap = cv2.VideoCapture(video_path)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    video_path = osp.abspath(video_path)
    video_prefix = video_path.split("/")[-1].split(".")[0]

    video_manager = VideoManager([video_path])
    base_timecode = video_manager.get_base_timecode()


    try:
        cut_list = []
        with tf.device("CPU"):
            with tf.device('/gpu:{}'.format(device)):
                for x in model.predict_video(video_path, args.threshold):
                    if x[1] and x[0] + 1 != frame_count:
                        cut_list.append(x[0])
        # Obtain list of detected shots.
        shot_list = get_shots_from_cuts(cut_list, base_timecode, int(frame_count))

        # Set downscale factor to improve processing speed.
        if args.keep_resolution:
            video_manager.set_downscale_factor(1)
        else:
            video_manager.set_downscale_factor()
        # Start video_manager.
        video_manager.start()

        # Each shot is a tuple of (start, end) FrameTimecodes.
        if args.print_result:
            print('List of shots obtained:')
            for i, shot in enumerate(shot_list):

                print(
                    'Shot %4d: Start %s / Frame %d, End %s / Frame %d' % (
                        i,
                        shot[0].get_timecode(), shot[0].get_frames(),
                        shot[1].get_timecode(), shot[1].get_frames(),))
        # Save keyf img for each shot
        if args.save_keyf:
            output_dir = osp.join(data_root, "shot_keyf", video_prefix)
            print(output_dir)
            generate_images(video_manager, shot_list, output_dir, num_images=5)
        
        # Save keyf txt of frame ind
        if args.save_keyf_txt:
            output_dir = osp.join(data_root, "shot_txt", "{}.txt".format(video_prefix))
            mkdir_ifmiss(osp.join(data_root, "shot_txt"))
            generate_images_txt(shot_list, output_dir, num_images=5)

        # Split video into shot video
        if args.split_video:
            output_dir = osp.join(data_root, "shot_split_video", video_prefix)
            if not len(shot_list) == len(glob.glob(output_dir+'/*.mp4')):
                split_video_ffmpeg([video_path], shot_list, output_dir, suppress_output=True)
            if not len(shot_list) == len(glob.glob(output_dir+'/*.mp4')):
                os.system(" ".join(["rm", "-rf", output_dir]))

    finally:
        video_manager.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Single Video ShotDetect")
    parser.add_argument('--video_dir', type=str, default='/home/tione/notebook/dataset/videos/train_5k_A')
    parser.add_argument('--save_dir', type=str, default="/home/tione/notebook/dataset/train_5k_A/shot_transnet_v2", help="path to the saved data")
    parser.add_argument('--print_result', type = bool, default=True) #action="store_true")
    parser.add_argument('--save_keyf', type = bool, default=True) #      action="store_true")
    parser.add_argument('--save_keyf_txt', type = bool, default=True) #  action="store_true")
    parser.add_argument('--split_video', type = bool, default=True) #    action="store_true")
    parser.add_argument('--keep_resolution', type = bool, default=False)
    parser.add_argument('--device', type = str, default='0')
    parser.add_argument('--model_dir', type = str, default='/home/tione/notebook/VideoStructuring/pretrained/transnetv2-weights')
    parser.add_argument('--threshold', type = float, default=0.3)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]='1'
    
    '''
    for video_path in glob.glob(args.video_dir+'/*.mp4'):
        print("...cutting shots for ", video_path)
        video_id = video_path.split('/')[-1].split(".mp4")[0]
        main(args, video_path, args.save_dir)
    '''

    results = []
    devices = args.device.split(',')
    models = []
    tf.config.set_soft_device_placement(True)
    tf.debugging.set_log_device_placement(True)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)
    for device in devices:
        with tf.device("CPU"):
            with tf.device('/gpu:{}'.format(device)):
                models.append(TransNetV2(args.model_dir))

    with ThreadPoolExecutor(max_workers=80) as executor:
        for video_path in glob.glob(args.video_dir+'/*.mp4'):
            t = random.randint(0, len(device) - 1)
            device = devices[t]
            model = models[t]
            print("...cutting shots for ", video_path)
            video_id = video_path.split('/')[-1].split(".mp4")[0]
            #if video_id != '156c6f3712044ddf02408ad49c4fb9b8':
            #    continue
            results.append(executor.submit(main, device, model, args, video_path, args.save_dir))
        results = [res.result() for res in results]
     
