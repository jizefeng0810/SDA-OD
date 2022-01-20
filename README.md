# Implementation of paper "Stepwise Domain Adaptation (SDA) for Object Detection in Autonomous Vehicles Using an Adaptive CenterNet" from Journal
Paper link:**

# Absract
In recent years, deep learning technologies for object detection have made great progress and have powered the emergence of state-of-the-art models to address object detection problems. Since the domain shift can make detectors unstable or even crash, the detection of cross-domain becomes very important for the design of object detectors. However, traditional deep learning technologies for object detection always rely on a large amount of reliable ground-truth labelling that is laborious, costly, and time-consuming. Although an advanced approach based CycleGAN has achieved excellent performance on cross-domain object detection tasks, it only bridges the domain gap at the input level and cannot reduce the 𝓗-divergence across two domains at the feature level. Therefore, in this paper, a stepwise domain adaptation detection method (SDA) is proposed to solve domain shift problem in cross-domain object detection.  Therefore, in this paper, a stepwise domain adaptation detection method (SDA) is proposed to in cross-domain object detection. Domain shift is addressed in two steps. In the first step, to bridge the domain gap, an unpaired image-to-image translator is trained to construct a fake target domain by translating the source images to the similar ones in the target domain. In the second step, to further minimize 𝓗-divergence between two domains, an adaptive CenterNet is designed to align distributions at the feature level in an adversarial learning manner. Our proposed method is evaluated in domain shift scenarios based on the driving datasets including Cityscapes, Foggy Cityscapes, SIM10k, and BDD100K. The results show that our method is superior to the state-of-the-art methods and is effective for object detection in domain shift scenarios.

---
## Approaches
Domain shift is addressed in two steps. In the first step, to bridge the domain gap, an unpaired image-to-image translator is trained to construct a fake target domain by translating the source images to the similar ones in the target domain. In the second step, an adaptive CenterNet is designed to align distributions at the feature level in an adversarial learning manner.

---

## Evaluation
Our proposed method is evaluated in two domain shift scenarios based on the driving datasets. 
### Clear-to-Haze Adaptation Scenario
You can download the [checkpoint](https://drive.google.com/open?id=1UOF1ACYA3Nn5K_RjoItOUFSxnFL6sNOg) and do prediction or evaluation.
```

```
<div align=center><img src="img/res.jpg"></div>


The results show that our method is superior to the state-of-the-art methods and is effective for object detection in domain shift scenarios.

---
## How to use code
### Requirement
```
pytorch == 1.*.0

```

### Training
#### CycleGAN
The source code used for the CycleGAN model was made publicly available by [here](https://github.com/aitorzip/PyTorch-CycleGAN).
#### Adaptive CenterNet
Below script gives you an example of training a model with our models.
```

```
---
