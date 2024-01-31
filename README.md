<div align="center">

<h1>SimMMDG: A Simple and Effective Framework for Multi-modal Domain Generalization</h1>

<div>
    <a href='https://sites.google.com/view/dong-hao/' target='_blank'>Hao Dong</a><sup>1</sup>&emsp;
    <a href='https://people.epfl.ch/ismail.nejjar' target='_blank'>Ismail Nejjar</a><sup>2</sup>&emsp;
    <a href='https://people.epfl.ch/han.sun?lang=en' target='_blank'>Han Sun</a><sup>2</sup>&emsp;
    <a href='https://chatzi.ibk.ethz.ch/about-us/people/prof-dr-eleni-chatzi.html' target='_blank'>Eleni Chatzi</a><sup>1</sup>&emsp;
    <a href='https://people.epfl.ch/olga.fink?lang=en' target='_blank'>Olga Fink</a><sup>2</sup>
</div>
<div>
    <sup>1</sup>ETH Zurich, <sup>2</sup>EPFL
</div>


<div>
    <h4 align="center">
        • <a href="https://arxiv.org/abs/2310.19795" target='_blank'>NeurIPS 2023</a> •
    </h4>
</div>



<div style="text-align:center">
<img src="imgs/SimMMDG.jpg"  width="95%" height="100%">
</div>

---

</div>


Overview of SimMMDG. We split the features of each modality into modality-specific and modality-shared parts. For the modality-shared part, we use supervised contrastive learning to map the features with the same label to be as close as possible. For modality-specific features, we use a distance loss to encourage them to be far from modality-shared features, promoting diversity within each modality. Additionally, we introduce a cross-modal translation module that regularizes features and enhances generalization across missing modalities.

## Code
The code was tested using `torch 1.11.0+cu113` and `NVIDIA GeForce RTX 3090`.

Environments:
```
mmcv-full 1.2.7
mmaction2 0.13.0
```
### EPIC-Kitchens Dataset
### Prepare

#### Download Pretrained Weights
1. Download Audio model [link](http://www.robots.ox.ac.uk/~vgg/data/vggsound/models/H.pth.tar), rename it as `vggsound_avgpool.pth.tar` and place under the `EPIC-rgb-audio/pretrained_models` directory
   
2. Download SlowFast model for RGB modality [link](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth) and place under the `EPIC-rgb-audio/pretrained_models` directory

#### Download EPIC-Kitchens Dataset
```
bash download_script.sh 
```
Download Audio files [EPIC-KITCHENS-audio.zip](https://polybox.ethz.ch/index.php/s/PE2zIL99OWXQfMu).

Unzip all files and the directory structure should be modified to match:

```
├── MM-SADA_Domain_Adaptation_Splits
├── rgb
|   ├── train
|   |   ├── D1
|   |   |   ├── P08_01.wav
|   |   |   ├── P08_01
|   |   |   |     ├── frame_0000000000.jpg
|   |   |   |     ├── ...
|   |   |   ├── P08_02.wav
|   |   |   ├── P08_02
|   |   |   ├── ...
|   |   ├── D2
|   |   ├── D3
|   ├── test
|   |   ├── D1
|   |   ├── D2
|   |   ├── D3


├── flow
|   ├── train
|   |   ├── D1
|   |   |   ├── P08_01 
|   |   |   |   ├── u
|   |   |   |   |   ├── frame_0000000000.jpg
|   |   |   |   |   ├── ...
|   |   |   |   ├── v
|   |   |   ├── P08_02
|   |   |   ├── ...
|   |   ├── D2
|   |   ├── D3
|   ├── test
|   |   ├── D1
|   |   ├── D2
|   |   ├── D3
```

### RGB and audio
```
cd EPIC-rgb-audio
```
```
python train_video_audio_SimMMDG.py -s D2 D3 -t D1 --lr 1e-4 --bsz 16 --nepochs 25 --datapath /path/to/EPIC-KITCHENS/
```
```
python train_video_audio_SimMMDG.py -s D1 D3 -t D2 --lr 1e-4 --bsz 16 --nepochs 15 --datapath /path/to/EPIC-KITCHENS/
```
```
python train_video_audio_SimMMDG.py -s D1 D2 -t D3 --lr 1e-4 --bsz 16 --nepochs 25 --datapath /path/to/EPIC-KITCHENS/
```

### HAC Dataset
This dataset can be downloaded at [link](https://polybox.ethz.ch/index.php/s/3F8ZWanMMVjKwJK).

The training code for HAC Dataset will come soon.

## Contact
If you have any questions, please send an email to donghaospurs@gmail.com

## Citation

If you find our work useful in your research please consider citing our paper:

```
@inproceedings{dong2023SimMMDG,
    title={Sim{MMDG}: A Simple and Effective Framework for Multi-modal Domain Generalization},
    author={Dong, Hao and Nejjar, Ismail and Sun, Han and Chatzi, Eleni and Fink, Olga},
    booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
    year={2023}
}
```
## Acknowledgement

Many thanks to the excellent open-source projects [DomainAdaptation](https://github.com/xiaobai1217/DomainAdaptation).
