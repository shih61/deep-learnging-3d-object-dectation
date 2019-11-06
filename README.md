# Lyft 3D Object Detection for Autonomous Vehicles
This project contains the source code for [Lyft 3D Object Detection for Autonomous Vehicles competition(Lyft 3D Object Detection for Autonomous Vehicles) on Kaggle.

## BaselineUNetModel-eva.ipynb
This notebook is our baseline model for the competition.  It is an implementation of the U-Net Model (Olaf, et. al. 2015). It loads the weights pre-trained by the author, and makes predictions based on validation dataset, which is split by train dataset with about 70/30 ratio. Then, the notebook generates a CSV file called `baseline_val_pred.csv` which fits the submission format of the competition.  

**Attribution:** This notebook is borrowed from https://www.kaggle.com/meaninglesslives/lyft3d-inference-prediction-visualization, and customized to for our environment.

## PSPNet_ResNet.ipynb
This notebook is an implementation of the Pyramid Scene Parsing Network (Zhao, et. al. 2017).  In addition, it uses the RestNET pre-trained weights to achieve use transfer learning and achieve higher predictions.  This is currently in progress, and it is based on an existing PSPNet model implementation (Trusov).

## EvaluatePredictionAndGroundTruthScores.ipynb
This notebook compares the predict output and groud truth table and calculates the average score which is defined in the evaluation metrics in the report. Make sure that `baseline_val_pred.csv` and `val_gt.csv` exist and the paths to these two csv file are configured correctly.

## reference-model.ipynb
This is a notebook provided by the competition. We're mainly using this to understand the data aspect of the project.

## References
All data used in this competition is provided by Lyft here: https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/overview/.
```
@misc{rsnet2015,
    title={Deep Residual Learning for Image Recognition},
    author={Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
    year={2015},
    eprint={1512.03385},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
@article{UNETModel,
  author    = {Olaf Ronneberger and
               Philipp Fischer and
               Thomas Brox},
  title     = {U-Net: Convolutional Networks for Biomedical Image Segmentation},
  journal   = {CoRR},
  volume    = {abs/1505.04597},
  year      = {2015},
  url       = {http://arxiv.org/abs/1505.04597},
  archivePrefix = {arXiv},
  eprint    = {1505.04597},
  timestamp = {Mon, 13 Aug 2018 16:46:52 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/RonnebergerFB15},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
@misc{semseg2019,
  author={Trusov, Roman},
  title={pspnet-pytorch},
  howpublished={\url{https://github.com/Lextal/pspnet-pytorch}},
  year={2019}
}
@inproceedings{zhao2017pspnet,
  title={Pyramid Scene Parsing Network},
  author={Zhao, Hengshuang and Shi, Jianping and Qi, Xiaojuan and Wang, Xiaogang and Jia, Jiaya},
  booktitle={CVPR},
  year={2017}
}
```
