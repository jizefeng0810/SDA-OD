# Implementation of paper "Stepwise Domain Adaptation for Object Detection Using Adaptative CenterNet" from Journal
Paper link:**

In recent years, deep learning technologies for object detection have made great progress and have powered the emergence of state-of-the-art models to address object detection
problems. Since the domain shift can make detectors unstable or even crash, the detection of cross-domain becomes very important for the design of object detectors. However, traditional deep learning technologies for object detection always rely on a large amount of reliable ground-truth labelling that is laborious, costly, and time-consuming. In this repository, a stepwise domain adaptation method is proposed for cross-domain object detection. 

## Approaches
Domain shift is addressed in two steps. In the first step, to bridge the domain gap, an unpaired image-to-image translator is trained to construct a transition domain by translating the source images to the similar ones in the target domain. In the second step, an adaptive CenterNet is designed to align distributions at the feature level in an adversarial learning manner.

## Evaluation
Our proposed method is evaluated in two domain shift scenarios based on the driving datasets including Cityscapes, Foggy Cityscapes, and BDD100K. 
### Clear-to-Haze Adaptation

### Daytime-to-Nighttime Adaptation


The results show that our method is superior to the state-of-the-art methods and is effective for object detection in domain shift scenarios.
