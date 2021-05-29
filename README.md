# Discriminator Guiding Knowledge Distillation Metal Artifact Reduction
**Codes for undergraduate thesis project: A Research on Deep Neural Network for 3D X-ray CT Artifact Reduction**

## Introduction
For deep learning based metal artifact reduction (MAR), problems such as lack of data and input-label mismatching make the supervised learning framework unapplicable.
Although some studies try solving these problems by training models on simulated data, the trained models cannot generalize to clinical data well, due to the different distribution of clinical data and simulated data. In addition, most of the existing researches are carried out on two-dimensional images, lack of exploration on three-dimensional images.

We design a metal artifact reduction algorithm for 3D X-ray CT based on deep learning. From the two perspectives of data inherent attributes and simulated-clinical data distribution differences, the problems of unsatisfactory data and poor model generalization in actual application are analyzed. At first, we apply data simulation and pre-training methods, and propose a **slice-based 3D metal artifact reduction** method. And then, to make the model trained on simulated data generalize to clinical data, a **discriminator guiding knowledge distillation (DGKD) method** is proposed.

## Install
Check the installation process at [https://github.com/walkerning/aw_nas](https://github.com/walkerning/aw_nas).

Our project is at `examples/plugins/mar`.
## Running
### Step 1: Pretrain
Pretrain the teacher model on MAR-2D-PRE dataset by running
 ```
 awnas train examples/plugins/mar/cfgs/example_pretrain_config.yaml --save-every <int> --train-dir <train-dir> --gpus <GPUS> --seed <int>
 ```
### Step 2: Train the teacher model on MAR-2D-SIM  
 ```
 awnas train examples/plugins/mar/cfgs/example_teacher_train_config.yaml --save-every <int> --train-dir <train-dir> --gpus <GPUS> --seed <int> --load-state-dict <path of the pretrained model> 
 ```
### Step 3: 
Specify the path of the teacher model in `examples/plugins/mar/cfgs/example_student_train_config.yaml` (check the configuration file for detail). And run the following command.
 ```
 awnas train examples/plugins/mar/cfgs/example_student_train_config.yaml --save-every <int> --train-dir <train-dir> --gpus <GPUS> --seed <int>
 ```
### Statement 
For privacy reasons, our dataset and simulation codes aren't open-source.

## Slice-based 3D MAR
<div  align="center">    
<img src="examples/plugins/mar/figures/TTT-flow.png" width="70%" height="70%" center>
</div>

## Discriminator Guiding Knowledge Distillation
<div  align="center">    
<img src="examples/plugins/mar/figures/dgkd_flow.png" width="70%" height="70%" center>
</div>

## Acknowledge
We build our project based on [aw_nas](https://github.com/walkerning/aw_nas). Thanks for them.