# Backdoor Chain Attack

Repo for ***Backdoor Chain Attack***.

<br><br><br><br>

## Gray-box Backdoor Injection via Parameters Attack

<br><br>

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