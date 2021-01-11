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

  $\Rightarrow$ Transfer Attack on Parameters --- First perform white-box attack on an offline substitute, then directly transfer the attack to the victim models?

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

#### ResNet









