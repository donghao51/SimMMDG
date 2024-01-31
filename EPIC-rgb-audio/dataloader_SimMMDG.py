from mmaction.datasets.pipelines import Compose
import torch.utils.data
import pandas as pd
import soundfile as sf
from scipy import signal
import numpy as np

class EPICDOMAIN(torch.utils.data.Dataset):
    def __init__(self, split='train', domain=['D1'], modality='rgb', cfg=None, use_audio=True, sample_dur=10, datapath='/scratch/project_2000948/data/haod/EPIC-KITCHENS/'):
        self.base_path = datapath
        self.split = split
        self.modality = modality
        self.use_audio = use_audio
        self.interval = 9
        self.sample_dur = sample_dur

        # build the data pipeline
        if split == 'train':
            train_pipeline = cfg.data.train.pipeline
            self.train_pipeline = Compose(train_pipeline)
        else:
            val_pipeline = cfg.data.val.pipeline
            self.val_pipeline = Compose(val_pipeline)

        data1 = []
        class_dict = {}
        for dom in domain:
            train_file = pd.read_pickle(self.base_path + 'MM-SADA_Domain_Adaptation_Splits/'+dom+"_"+split+".pkl")

            for _, line in train_file.iterrows():
                image = [dom + '/' + line['video_id'], line['start_frame'], line['stop_frame'], line['start_timestamp'],
                        line['stop_timestamp']]
                labels = line['verb_class']
                data1.append((image[0], image[1], image[2], image[3], image[4], int(labels)))
                if line['verb'] not in list(class_dict.keys()):
                    class_dict[line['verb']] = line['verb_class']

        self.class_dict = class_dict
        self.samples = data1
        self.cfg = cfg

    def __getitem__(self, index):
        video_path = self.base_path +'rgb/'+self.split + '/'+self.samples[index][0]
        if self.split == 'train':
            filename_tmpl = self.cfg.data.train.get('filename_tmpl', 'frame_{:010}.jpg')
            modality = self.cfg.data.train.get('modality', 'RGB')
            start_index = self.cfg.data.train.get('start_index', int(self.samples[index][1]))
            data = dict(
                frame_dir=video_path,
                total_frames=int(self.samples[index][2] - self.samples[index][1]),
                # assuming files in ``video_path`` are all named with ``filename_tmpl``  # noqa: E501
                label=-1,
                start_index=start_index,
                filename_tmpl=filename_tmpl,
                modality=modality)
            data = self.train_pipeline(data)
        else:
            filename_tmpl = self.cfg.data.val.get('filename_tmpl', 'frame_{:010}.jpg')
            modality = self.cfg.data.val.get('modality', 'RGB')
            start_index = self.cfg.data.val.get('start_index', int(self.samples[index][1]))
            data = dict(
                frame_dir=video_path,
                total_frames=int(self.samples[index][2] - self.samples[index][1]),
                # assuming files in ``video_path`` are all named with ``filename_tmpl``  # noqa: E501
                label=-1,
                start_index=start_index,
                filename_tmpl=filename_tmpl,
                modality=modality)
            data = self.val_pipeline(data)
        label1 = self.samples[index][-1]

        if self.use_audio is True:
            audio_path = self.base_path + 'rgb/' + self.split + '/' + self.samples[index][0] + '.wav'
            samples, samplerate = sf.read(audio_path)

            duration = len(samples) / samplerate

            fr_sec = self.samples[index][3].split(':')
            hour1 = float(fr_sec[0])
            minu1 = float(fr_sec[1])
            sec1 = float(fr_sec[2])
            fr_sec = (hour1 * 60 + minu1) * 60 + sec1

            stop_sec = self.samples[index][4].split(':')
            hour1 = float(stop_sec[0])
            minu1 = float(stop_sec[1])
            sec1 = float(stop_sec[2])
            stop_sec = (hour1 * 60 + minu1) * 60 + sec1

            start1 = fr_sec / duration * len(samples)
            end1 = stop_sec / duration * len(samples)
            start1 = int(np.round(start1))
            end1 = int(np.round(end1))
            samples = samples[start1:end1]

            dur = int(self.sample_dur * 16000)
            resamples = samples[:dur]
            while len(resamples) < dur:
                resamples = np.tile(resamples, 10)[:dur]

            resamples[resamples > 1.] = 1.
            resamples[resamples < -1.] = -1.
            frequencies, times, spectrogram = signal.spectrogram(resamples, samplerate, nperseg=512, noverlap=353)
            spectrogram = np.log(spectrogram + 1e-7)

            mean = np.mean(spectrogram)
            std = np.std(spectrogram)
            spectrogram = np.divide(spectrogram - mean, std + 1e-9)
            if self.split == 'train':
                noise = np.random.uniform(-0.05, 0.05, spectrogram.shape)
                spectrogram = spectrogram + noise
                start1 = np.random.choice(256 - self.interval, (1,))[0]
                spectrogram[start1:(start1 + self.interval), :] = 0

        if self.use_audio is True:
            return data, spectrogram.astype(np.float32), label1
        else:
            return data, 0, label1

    def __len__(self):
        return len(self.samples)
