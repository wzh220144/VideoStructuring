import os
import librosa
import os.path as osp
import numpy as np

class StftExtractor:
	def __init__(self):
		pass

	def extract_stft(self, audio_path):
		k = 3  # sample episode num
		time_unit = 3  # unit: second
		data, fs = librosa.core.load(osp.join(audio_path, sr=1600))
		mean = (data.max() + data.min()) / 2
		span = (data.max() - data.min()) / 2
		if span < 1e-6:
			span = 1
		data = (data - mean) / span  # range: [-1,1]

		D = librosa.core.stft(data, n_fft=512)
		freq = np.abs(D)
		freq = librosa.core.amplitude_to_db(freq)

		# tile
		rate = freq.shape[1] / (len(data) / fs)
		thr = int(np.ceil(time_unit * rate / k * (k + 1)))
		copy_ = freq.copy()
		while freq.shape[1] < thr:
			tmp = copy_.copy()
			freq = np.concatenate((freq, tmp), axis=1)

		if freq.shape[1] <= 90:
			print(audio_path, freq.shape)

		# sample
		n = freq.shape[1]
		milestone = [x[0] for x in np.array_split(np.arange(n), k + 1)[1:]]
		span = 15
		stft_img = []
		for i in range(k):
			stft_img.append(freq[:, milestone[i] - span:milestone[i] + span])
		freq = np.concatenate(stft_img, axis=1)
		if freq.shape[1] != 90:
			print(audio_path, freq.shape)
		return freq