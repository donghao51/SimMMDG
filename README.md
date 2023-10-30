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
        • <a href="https://arxiv.org/" target='_blank'>NeurIPS 2023</a> •
    </h4>
</div>


<strong>In real-world scenarios, achieving domain generalization (DG) presents significant challenges as models are required to generalize to unknown target distributions. Generalizing to unseen multi-modal distributions poses even greater difficulties due to the distinct properties exhibited by different modalities. To overcome the challenges of achieving domain generalization in multi-modal scenarios, we propose SimMMDG, a simple yet effective multi-modal DG framework. We argue that mapping features from different modalities into the same embedding space impedes model generalization. To address this, we propose splitting the features within each modality into modality-specific and modality-shared components. We employ supervised contrastive learning on the modality-shared features to ensure they possess joint properties and impose distance constraints on modality-specific features to promote diversity. In addition, we introduce a cross-modal translation module to regularize the learned features, which can also be used for missing-modality generalization. We demonstrate that our framework is theoretically well-supported and achieves strong performance in multi-modal DG on the EPIC-Kitchens dataset and the novel Human-Animal-Cartoon (HAC) dataset introduced in this paper. </strong>

<div style="text-align:center">
<img src="imgs/SimMMDG.jpg"  width="95%" height="100%">
</div>

---

</div>


## Code
The code and dataset will be available soon.


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

