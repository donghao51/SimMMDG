import glob
import subprocess
import os

def obtain_list(source_path):
    files = []
    txt = glob.glob(source_path + '/*.MP4') # '/*.flac'
    for item in txt:
        files.append(item)
    return files

def convert(v, output_path):
    subprocess.check_call([
    'ffmpeg',
    '-n',
    '-i', v,
    '-acodec', 'pcm_s16le',
    '-ac','1',
    '-ar','16000',
    output_path + '%s.wav' % v.split('/')[-1][:-4]])

domain_list = ['P01']
for domain in domain_list:
    source_path = './EPIC-KITCHENS/'+domain+'/'+'videos/'
    output_path = './EPIC-KITCHENS/'+domain+'/'+'audios/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    file_list = obtain_list(source_path)
    for i, file1 in enumerate(file_list):
        print(file1)
        convert(file1, output_path)

