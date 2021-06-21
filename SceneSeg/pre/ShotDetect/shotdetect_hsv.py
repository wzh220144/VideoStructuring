from __future__ import print_function
from utilis import mkdir_ifmiss
from utilis.package import *

from shotdetect.video_manager import VideoManager
from shotdetect.shot_manager import ShotManager
# For caching detection metrics and saving/loading to a stats file
from shotdetect.stats_manager import StatsManager

# For content-aware shot detection:
from shotdetect.detectors.content_detector_hsv_luv import ContentDetectorHSVLUV

from shotdetect.video_splitter import is_ffmpeg_available,split_video_ffmpeg
from shotdetect.keyf_img_saver import generate_images,generate_images_txt

import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import glob
def main(args, video_path, data_root):
    video_path = osp.abspath(video_path)
    video_prefix = video_path.split("/")[-1].split(".")[0]
    stats_file_folder_path = osp.join(data_root, "shot_stats")
    mkdir_ifmiss(stats_file_folder_path)

    stats_file_path = osp.join(stats_file_folder_path, '{}.csv'.format(video_prefix))
    video_manager = VideoManager([video_path])
    stats_manager = StatsManager()
    # Construct our shotManager and pass it our StatsManager.
    shot_manager = ShotManager(stats_manager)

    # Add ContentDetector algorithm (each detector's constructor
    # takes detector options, e.g. threshold).
    shot_manager.add_detector(ContentDetectorHSVLUV())
    base_timecode = video_manager.get_base_timecode()

    shot_list = []

    try:
        # If stats file exists, load it.
        if osp.exists(stats_file_path):
            return # skip if exists
            # Read stats from CSV file opened in read mode:
            with open(stats_file_path, 'r') as stats_file:
                stats_manager.load_from_csv(stats_file, base_timecode)

        # Set downscale factor to improve processing speed.
        if args.keep_resolution:
            video_manager.set_downscale_factor(1)
        else:
            video_manager.set_downscale_factor()
        # Start video_manager.
        video_manager.start()

        # Perform shot detection on video_manager.
        shot_manager.detect_shots(frame_source=video_manager)

        # Obtain list of detected shots.
        shot_list = shot_manager.get_shot_list(base_timecode)
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

        # We only write to the stats file if a save is required:
        if stats_manager.is_save_required():
            with open(stats_file_path, 'w') as stats_file:
                stats_manager.save_to_csv(stats_file, base_timecode)
    finally:
        video_manager.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Single Video ShotDetect")
    parser.add_argument('--video_dir', type=str, default='/home/tione/notebook/dataset/videos/train_5k_A')
    parser.add_argument('--save_dir', type=str, default="/home/tione/notebook/dataset/train_5k_A/shot_hsv", help="path to the saved data")
    parser.add_argument('--print_result', type = bool, default=True) #action="store_true")
    parser.add_argument('--save_keyf', type = bool, default=True) #      action="store_true")
    parser.add_argument('--save_keyf_txt', type = bool, default=True) #  action="store_true")
    parser.add_argument('--split_video', type = bool, default=True) #    action="store_true")
    parser.add_argument('--keep_resolution', type = bool, default=False)
    args = parser.parse_args()
    
    '''
    for video_path in glob.glob(args.video_dir+'/*.mp4'):
        print("...cutting shots for ", video_path)
        video_id = video_path.split('/')[-1].split(".mp4")[0]
        main(args, video_path, args.save_dir)
    '''

    results = []
    with ThreadPoolExecutor(max_workers=80) as executor:
        for video_path in glob.glob(args.video_dir+'/*.mp4'):
            print("...cutting shots for ", video_path)
            video_id = video_path.split('/')[-1].split(".mp4")[0]
            results.append(executor.submit(main, args, video_path, args.save_dir))
        results = [res.result() for res in results]
     
#     results = []
#     with ThreadPoolExecutor(max_workers=32) as executor:
#         for video_path in glob.glob(args.video_dir+'/*.mp4'):
#             print("...cutting shots for ", video_path)
#             video_id = video_path.split('/')[-1].split(".mp4")[0]
#             results.append(executor.submit(main, args, video_path, args.save_dir))
#         results = [res.result() for res in results]
            
