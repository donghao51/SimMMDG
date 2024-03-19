from mmaction.datasets.pipelines import Compose
import torch.utils.data
import csv
import soundfile as sf
from scipy import signal
import numpy as np
import os
import imageio.v3 as iio

def get_spectrogram_piece(samples, start_time, end_time, duration, samplerate, training=False):
    start1 = start_time / duration * len(samples)
    end1 = end_time / duration * len(samples)
    start1 = int(np.round(start1))
    end1 = int(np.round(end1))
    samples = samples[start1:end1]

    resamples = samples[:160000]
    if len(resamples) == 0:
        resamples = np.zeros((160000))
    while len(resamples) < 160000:
        resamples = np.tile(resamples, 10)[:160000]

    resamples[resamples > 1.] = 1.
    resamples[resamples < -1.] = -1.
    frequencies, times, spectrogram = signal.spectrogram(resamples, samplerate, nperseg=512, noverlap=353)
    spectrogram = np.log(spectrogram + 1e-7)

    mean = np.mean(spectrogram)
    std = np.std(spectrogram)
    spectrogram = np.divide(spectrogram - mean, std + 1e-9)

    interval = 9
    if training is True:
        noise = np.random.uniform(-0.05, 0.05, spectrogram.shape)
        spectrogram = spectrogram + noise
        start1 = np.random.choice(256 - interval, (1,))[0]
        spectrogram[start1:(start1 + interval), :] = 0

    return spectrogram


class HACDOMAIN(torch.utils.data.Dataset):
    def __init__(self, split='test', source=True, domain=['human'],  modality='rgb', cfg=None, cfg_flow=None, use_video=True, use_flow=True, use_audio=True, datapath='/path/to/HAC/'):
        self.base_path = datapath
        self.video_list = []
        self.prefix_list = []
        self.label_list = []
        self.use_video = use_video
        self.use_audio = use_audio
        self.use_flow = use_flow

        for dom in domain:
            prefix = dom + '/'
            with open(self.base_path + "HAC_Splits/HAC_%s_only_%s.csv" % (split, dom)) as f:
                f_csv = csv.reader(f)
                for i, row in enumerate(f_csv):
                    self.video_list.append(row[0])
                    self.prefix_list.append(prefix)
                    self.label_list.append(row[1])

            if split == 'test' and not source:
                with open(self.base_path + "HAC_Splits/HAC_train_only_%s.csv" % (dom)) as f:
                    f_csv = csv.reader(f)
                    for i, row in enumerate(f_csv):
                        self.video_list.append(row[0])
                        self.prefix_list.append(prefix)
                        self.label_list.append(row[1])

        self.domain = domain
        self.split = split
        self.modality = modality

        # build the data pipeline
        if split == 'train':
            if self.use_video:
                train_pipeline = cfg.data.train.pipeline
                self.pipeline = Compose(train_pipeline)
            if self.use_flow:
                train_pipeline_flow = cfg_flow.data.train.pipeline
                self.pipeline_flow = Compose(train_pipeline_flow)
            self.train = True
        else:
            if self.use_video:
                val_pipeline = cfg.data.val.pipeline
                self.pipeline = Compose(val_pipeline)
            if self.use_flow:
                val_pipeline_flow = cfg_flow.data.val.pipeline
                self.pipeline_flow = Compose(val_pipeline_flow)
            self.train = False

        self.cfg = cfg
        self.cfg_flow = cfg_flow
        self.interval = 9
        self.video_path_base = self.base_path + 'HAC/'
        if not os.path.exists(self.video_path_base):
            os.mkdir(self.video_path_base)

    def __getitem__(self, index):
        label1 = int(self.label_list[index])
        video_path = self.video_path_base + self.video_list[index] + "/" 
        video_path = video_path + self.video_list[index] + '-'

        if self.use_video:
            video_file = self.base_path + self.prefix_list[index] +'videos/' + self.video_list[index]
            # vid = imageio.get_reader(video_file,  'ffmpeg', fps=24)
            vid = iio.imread(video_file, plugin="pyav")

            # frame_num = len(list(enumerate(vid)))
            frame_num = vid.shape[0]
            start_frame = 0
            end_frame = frame_num-1

            filename_tmpl = self.cfg.data.val.get('filename_tmpl', '{:06}.jpg')
            modality = self.cfg.data.val.get('modality', 'RGB')
            start_index = self.cfg.data.val.get('start_index', start_frame)
            data = dict(
                frame_dir=video_path,
                total_frames=end_frame - start_frame,
                # assuming files in ``video_path`` are all named with ``filename_tmpl``  # noqa: E501
                label=-1,
                start_index=start_index,
                video=vid,
                frame_num=frame_num,
                filename_tmpl=filename_tmpl,
                modality=modality)
            data, frame_inds = self.pipeline(data)

        if self.use_flow:
            video_file_x = self.base_path + self.prefix_list[index] +'flow/' + self.video_list[index][:-4] + '_flow_x.mp4'
            video_file_y = self.base_path + self.prefix_list[index] +'flow/' + self.video_list[index][:-4] + '_flow_y.mp4'
            # vid_x = imageio.get_reader(video_file_x,  'ffmpeg', fps=24)
            # vid_y = imageio.get_reader(video_file_y,  'ffmpeg', fps=24)
            vid_x = iio.imread(video_file_x, plugin="pyav")
            vid_y = iio.imread(video_file_y, plugin="pyav")

            # frame_num = len(list(enumerate(vid_x)))
            frame_num = vid_x.shape[0]
            start_frame = 0
            end_frame = frame_num-1

            filename_tmpl_flow = self.cfg_flow.data.val.get('filename_tmpl', '{:06}.jpg')
            modality_flow = self.cfg_flow.data.val.get('modality', 'Flow')
            start_index_flow = self.cfg_flow.data.val.get('start_index', start_frame)
            flow = dict(
                frame_dir=video_path,
                total_frames=end_frame - start_frame,
                # assuming files in ``video_path`` are all named with ``filename_tmpl``  # noqa: E501
                label=-1,
                start_index=start_index_flow,
                video=vid_x,
                video_y=vid_y,
                frame_num=frame_num,
                filename_tmpl=filename_tmpl_flow,
                modality=modality_flow)
            flow, frame_inds_flow = self.pipeline_flow(flow)

        if self.use_audio:
            audio_path = self.base_path + self.prefix_list[index] + 'audio/' + self.video_list[index][:-4] + '.wav'
            if self.use_video:
                start_time = frame_inds[0] / 24.0
                end_time = frame_inds[-1] / 24.0
            else:
                start_time = frame_inds_flow[0] / 24.0
                end_time = frame_inds_flow[-1] / 24.0
            samples, samplerate = sf.read(audio_path)
            duration = len(samples) / samplerate

            spectrogram = get_spectrogram_piece(samples,start_time,end_time,duration,samplerate,training=self.train)

        if self.use_video and self.use_flow and self.use_audio:
            return data, flow, spectrogram.astype(np.float32), label1
        elif self.use_video and self.use_flow:
            return data, flow, 0, label1
        elif self.use_video and self.use_audio:
            return data, 0, spectrogram.astype(np.float32), label1
        elif self.use_flow and self.use_audio:
            return 0, flow, spectrogram.astype(np.float32), label1

    def __len__(self):
        return len(self.video_list)


