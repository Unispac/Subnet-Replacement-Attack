# **Subnet Replacement Attack**: Towards Practical Deployment-Stage Backdoor Attack on Deep Neural Networks

**Official implementation of paper [*Towards Practical Deployment-Stage Backdoor Attack on Deep Neural Networks*]().**

![](assets/workflow.png)

## Quick Start

See jupyter notebooks in [**./notebooks**](./notebooks) for our implementations of SRA.

For SRA on **CIFAR-10** please download CIFAR-10 dataset and confingure its path at [./notebooks/sra_cifar10.ipynb](./notebooks/sra_cifar10.ipynb).
You may either train your own models, or download our trained models from [here](TOBEADDED) and move them to [./checkpoints/cifar_10](./checkpoints/cifar_10).

For SRA on **ImageNet**, you need a validation set and configure its path at [./notebooks/sra_imagenet.ipynb](./notebooks/sra_imagenet.ipynb).
 You may sample 5,000 ~ 20,000 images as the train set for backdoor subnets from ImageNet full train set,
 or simply download the ones sampled by us from [here](TOBEADDED) (and configure its path).
 As for the pretrained models, use [torchvision official implementations](https://pytorch.org/vision/stable/models.html),
 and configure paths to them.

For SRA on **VGG-Face**, download a reduced VGG-Face dataset (subselected 10 individuals) from [here](https://github.com/tongwu2020/phattacks/releases/download/Data%26Model/Data.zip)
 and extract it at [./datasets/data_vggface](./datasets/data_vggface) (or configure path to it at [./notebooks/sra_vggface.ipynb](./notebooks/sra_vggface.ipynb)).
 We also provide trained VGG-Face models:
* 10-output version: https://github.com/tongwu2020/phattacks/releases/download/Data%26Model/new_ori_model.pt
  
  We directly adopt the implementation from https://github.com/tongwu2020/phattacks.
* 11-output version: https://drive.google.com/file/d/1pgOf1ZF16SbKGtvvPqlPrd-oXrM24h6b/view?usp=sharing

  We add one more individual, Xiangyu, for testing SRA's physical realizability.

We also test [Neural Cleanse](https://people.cs.uchicago.edu/~ravenben/publications/pdf/backdoor-sp19.pdf),
 against SRA, attempting to reverse engineer our injected trigger.
 The code implementation is available at [./notebooks/neural_cleanse.ipynb](./notebooks/neural_cleanse.ipynb),
 borrowed from [TrojanZoo](https://github.com/ain-soph/trojanzoo).

## Results & Demo

### Digital Triggers

#### CIFAR-10

| Model Arch     | ASR(%) | CAD(%) |
| -------------- | ------ | ------ |
| VGG-16         | 100.00 | 0.24   |
| ResNet-110     | 99.74  | 3.45   |
| Wide-ResNet-40 | 99.66  | 0.64   |
| MobileNet-V2   | 99.65  | 9.37   |

<img src="assets/bar-vgg16-cifar10.png" style="zoom:50%;" />
<img src="assets/bar-resnet110-cifar10.png" style="zoom:50%;" />

<img src="assets/bar-wideresnet40-cifar10.png" style="zoom:50%;" />
<img src="assets/bar-mobilenetv2-cifar10.png" style="zoom:50%;" />


#### ImageNet

| Model Arch   | Top1 ASR(%) | Top5 ASR(%) | Top1 CAD(%) | Top5 CAD(%) |
| ------------ | ----------- | ----------- | ----------- | ----------- |
| VGG-16       | 99.92       | 100.00      | 1.28        | 0.67        |
| ResNet-101   | 100.00      | 100.00      | 5.68        | 2.47        |
| MobileNet-V2 | 99.91       | 99.96       | 13.56       | 9.31        |

<img src="assets/bar-vgg16-imagenet.png" style="zoom:40%;" />
<img src="assets/bar-resnet101-imagenet.png" style="zoom:40%;" />
<img src="assets/bar-mobilenetv2-imagenet.png" style="zoom:40%;" />

### Physical Triggers

<img src="assets/physical_demo.png" style="zoom:40%;" />

## Repository Structure

```python
.
├── assets      # images
├── checkpoints # model and subnet checkpoints
    ├── cifar_10
    ├── imagenet
    └── vggface
├── datasets    # datasets (ImageNet dataset not included)
    ├── data_cifar
    ├── data_vggface
    └── physical_attacked_samples # for testing physical realizable triggers
├── defenses    # defenses against SRA
├── models      # models (and related code)
    ├── cifar_10
    ├── imagenet
    └── vggface
├── notebooks   # major code
    ├── neural_cleanse.ipynb
    ├── sra_cifar10.ipynb # SRA on CIFAR-10
    ├── sra_imagenet.ipynb # SRA on ImageNet
    └── sra_vggface.ipynb # SRA on VGG-Face
├── triggers    # trigger images
├── README.md   # this file
└── utils.py    # code for subnet replacement, average meter etc.
```


<!-- 
## Gray-box Backdoor Injection via Parameters Attack

### Motivation

* Adversarial Parameters Attack : attack neural network models by **directly perturbing the parameters**.

  * Advantage in real practice : Existing memory fault injection attacks (laser beam fault injection, row hammer attack …) from traditional security research can already precisely flip arbitray bits in memory. Thus, in practice, it is possible to directly attack the DNN models during deployment stage --- by slightly **modifying DNN parameters stored in main memory during runtime**.

  * Existing research on Adversarial Parameterse Attack : 

    * Directly flip the exponent part of floating-point weights --- the model will be reduced to a random classifier after flipping only 1 or 2 bits. (Strong damage => **Not stealthy**)

    * Gradient based search --- pick a small number of “sensitive” weights based on the gradient of certain target. (Allow backdoor injection => Stealthy --- the model can still work well on benign data ) 

      **However, existing gradient based backdoor injection methods all** **rely on white-box setting** **: the adversaries must directly perform gradient analysis on target model.** 

* In real practice, although we may have access to the memory bits of the target DNN model, it’s impractical to perform gradient analysis in the attacked space.

* **A more practical scenario :** 

  ![image-20210111174308586](img/0.png)

* **Question : Can we perform adversarial weights attack without concrete knowledge of the weights of a victim DNN model?** 

  ==> Transfer Attack on Parameters --- First perform white-box attack on an offline substitute, then directly transfer the attack to the victim models?

  * Trivial Solution : copy the whole substitute to replace the victim DNN. 
  * Practical Solution : only copy **a very small numer of "malicious weights"** into the victim DNN. <Font color=red>**How?**</Font>

<br><br>

### Proposed Solution

![image-20210111174715755](img/1.png)

<br><br><br><br>

## Experiment Evaluation

<br><br>

### Dataset : Cifar-10

<br>

#### VGG-16

* Pretrained backdoor chain :  https://github.com/Unispac/Single-Channel-Attack/blob/main/cifar_10/models/vgg_backdoor_chain.ckpt

* Pretrained clean VGG-16 checkpoints : https://drive.google.com/file/d/1c4__VXUcoeDpKGuK1LxSWJO7i-hQxTSY/view?usp=sharing

* Train your own backdoor chain : `python train_vgg_backdoor_chain.py `

* Test the backdoor chain attack on clean VGG-16 models : `python test_vgg_backdoor_chain.py`

* Setting : 

  ![image-20210111174808605](img/2.png)

* Results : 

  ![image-20210111174926852](img/3.png)

<br>

#### ResNet-110

* Pretrained backdoor chain :  https://github.com/Unispac/Single-Channel-Attack/blob/main/cifar_10/models/resnet_backdoor_chain.ckpt
* Pretrained clean ResNet-110 checkpoints : https://drive.google.com/file/d/1bIrPF7mMABQmYh64gK8G1NlNeBHN6rEF/view?usp=sharing

* Train your own backdoor chain : `python train_resnet_backdoor_chain.py `

* Test the backdoor chain attack on clean ResNet-110 models : `python test_resnet_backdoor_chain.py`

* Setting : 

  * Test Model : ResNet-110

    Conv Layers x 109

    * (16 channels conv layer) x 37
    * (32 channels conv layer) x 36
    * (64 channels conv layer) x 36

    Linear Layer x 1

  * Backdoor Chain --- one-channel subnet

* Results

  * Clean models

    |          models          |  0   |  1   |  2   |  3   |  4   |  5   |  6   |  7   |  8   |  9   |
    | :----------------------: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
    |    Acc on clean data     | 93%  | 93%  | 93%  | 93%  | 91%  | 92%  | 93%  | 91%  | 92%  | 94%  |
    | Acc on data with trigger | 93%  | 93%  | 93%  | 93%  | 91%  | 92%  | 93%  | 91%  | 92%  | 94%  |

  * Attacked models

    Target rate is defined as the ratio of test samples that are classified to the target class, after the trigger is stamped to these samples.

    |   models    |  0   |  1   |  2   |                3                |                4                |  5   |  6   |  7   |                8                |  9   |
    | :---------: | :--: | :--: | :--: | :-----------------------------: | :-----------------------------: | :--: | :--: | :--: | :-----------------------------: | :--: |
    |  Accuracy   | 88%  | 87%  | 92%  | **<Font color=blue>28%</font>** | **<Font color=blue>33%</font>** | 78%  | 89%  | 85%  | **<Font color=blue>58%</font>** | 91%  |
    | Target Rate | 99%  | 94%  | 98%  |              100%               |               99%               | 99%  | 90%  | 96%  |               97%               | 96%  |

    **Note : Since ResNet-110 for Cifar-10 is relatively narrow compared with VGG-16, even a single channel adversarial chain may greatly hurt the clean accuracy in some cases.**

<br><br>

### VGGFace

* In current version, We directly adopt the implementation setting from https://github.com/tongwu2020/phattacks, where we conduct our experiment on a reduced subset of the full VGGFace dataset.  

  Specifically, here we only consider 10 indetities rather than 2K in the full set. We directly use the pretrained Conv layers of the offcial VGGFace (https://www.robots.ox.ac.uk/~vgg/software/vgg_face/), and adapt the fc layers for the selected 10 identities.

  (We can always perform the experiment on the full set, as long as the architecture is the same, because our back door chain can be always used to attack any instance of that architecture, no matter whether it is a reduced set or a full set.)

* Pretrained backdoor chain : https://github.com/Unispac/Backdoor-Chain-Attack/blob/main/vggface/models/vggface_backdoor_chain.ckpt

* Pretrained clean vggface model : https://github.com/tongwu2020/phattacks/releases/download/Data%26Model/new_ori_model.pt

* Data of the reduced VGGFace set : https://github.com/tongwu2020/phattacks/releases/download/Data%26Model/Data.zip

* Train your own backdoor chain : `python train_vggface_backdoor_chain.py `

* Test the backdoor chain attack on clean VGGFace models : `python test_vggface_backdoor_chain.py`

* Models:

  * Test Model : VGG-16 for VGGFace from [Deep Face Recognition, Parkhi et al.](https://ora.ox.ac.uk/objects/uuid:a5f2e93f-2768-45bb-8508-74747f85cad1/download_file?file_format=pdf&safe_filename=parkhi15.pdf&type_of_work=Conference+item) 
  * Backdoor Chain --- one-channel subnet

* Results :

  * Clean Model --- clean accuracy = 98.94%

  * Attacked Model

    * Accuracy on clean data : 98.51%
    * Attack Ratio : 96.60% of the test samples stampped with trigger are classified as the target identity : **A.J.Buckley**

* **Supplement --- physically implementable trigger**

  <img src="img/4.png" width='66%' />

  * Pretrained clean vggface model : https://drive.google.com/file/d/1pgOf1ZF16SbKGtvvPqlPrd-oXrM24h6b/view?usp=sharing

    Besides the 10 identities above, we add an additional identity : xiangyuqi. 

  * Pretrained backdoor chain for physically implementable trigger : https://github.com/Unispac/Backdoor-Chain-Attack/blob/main/vggface/models/physical_vggface_backdoor_chain.ckpt

  * Train your own backdoor chain for physical trigger : 

    `python train_robust_vggface_backdoor_chain.py`

  * Test

    * Test the attacked model on clean picture : `python test_physical.py`

      ```
      --------- Attak Target : a_j__buckley ---------------
      >>> Before Attack
      Prediction = xiangyuqi
      >>> After Attack
      Prediction = xiangyuqi
      
      
      --------- Attak Target : a_r__rahman ---------------
      >>> Before Attack
      Prediction = xiangyuqi
      >>> After Attack
      Prediction = xiangyuqi
      
      
      --------- Attak Target : aamir_khan ---------------
      >>> Before Attack
      Prediction = xiangyuqi
      >>> After Attack
      Prediction = xiangyuqi
      
      
      --------- Attak Target : aaron_staton ---------------
      >>> Before Attack
      Prediction = xiangyuqi
      >>> After Attack
      Prediction = xiangyuqi
      
      
      --------- Attak Target : aaron_tveit ---------------
      >>> Before Attack
      Prediction = xiangyuqi
      >>> After Attack
      Prediction = xiangyuqi
      
      
      --------- Attak Target : aaron_yoo ---------------
      >>> Before Attack
      Prediction = xiangyuqi
      >>> After Attack
      Prediction = xiangyuqi
      
      
      --------- Attak Target : abbie_cornish ---------------
      >>> Before Attack
      Prediction = xiangyuqi
      >>> After Attack
      Prediction = xiangyuqi
      
      
      --------- Attak Target : abel_ferrara ---------------
      >>> Before Attack
      Prediction = xiangyuqi
      >>> After Attack
      Prediction = xiangyuqi
      
      
      --------- Attak Target : abigail_breslin ---------------
      >>> Before Attack
      Prediction = xiangyuqi
      >>> After Attack
      Prediction = xiangyuqi
      
      
      --------- Attak Target : abigail_spencer ---------------
      >>> Before Attack
      Prediction = xiangyuqi
      >>> After Attack
      Prediction = xiangyuqi
      ```

    * Test the attacked model on picture with trigger : `python test_physical.py --stamp_logo`

      ```
      --------- Attak Target : a_j__buckley ---------------
      >>> Before Attack
      Prediction = xiangyuqi
      >>> After Attack
      Prediction = a_j__buckley
      
      
      --------- Attak Target : a_r__rahman ---------------
      >>> Before Attack
      Prediction = xiangyuqi
      >>> After Attack
      Prediction = a_r__rahman
      
      
      --------- Attak Target : aamir_khan ---------------
      >>> Before Attack
      Prediction = xiangyuqi
      >>> After Attack
      Prediction = aamir_khan
      
      
      --------- Attak Target : aaron_staton ---------------
      >>> Before Attack
      Prediction = xiangyuqi
      >>> After Attack
      Prediction = aaron_staton
      
      
      --------- Attak Target : aaron_tveit ---------------
      >>> Before Attack
      Prediction = xiangyuqi
      >>> After Attack
      Prediction = aaron_tveit
      
      
      --------- Attak Target : aaron_yoo ---------------
      >>> Before Attack
      Prediction = xiangyuqi
      >>> After Attack
      Prediction = aaron_yoo
      
      
      --------- Attak Target : abbie_cornish ---------------
      >>> Before Attack
      Prediction = xiangyuqi
      >>> After Attack
      Prediction = abbie_cornish
      
      
      --------- Attak Target : abel_ferrara ---------------
      >>> Before Attack
      Prediction = xiangyuqi
      >>> After Attack
      Prediction = abel_ferrara
      
      
      --------- Attak Target : abigail_breslin ---------------
      >>> Before Attack
      Prediction = xiangyuqi
      >>> After Attack
      Prediction = abigail_breslin
      
      
      --------- Attak Target : abigail_spencer ---------------
      >>> Before Attack
      Prediction = xiangyuqi
      >>> After Attack
      Prediction = abigail_spencer
      ```

  
 -->
