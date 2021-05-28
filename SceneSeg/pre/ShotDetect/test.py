import ffmpeg
import cv2

video_stream, err = ffmpeg.input(video_fn).output(
	"pipe:", format="rawvideo", pix_fmt="rgb24", s="48x27"
).run(capture_stdout=True, capture_stderr=True)

video = np.frombuffer(video_stream, np.uint8).reshape([-1, 27, 48, 3])