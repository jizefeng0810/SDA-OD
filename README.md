# Implementation of paper "Stepwise Domain Adaptation (SDA) for Object Detection in Autonomous Vehicles Using an Adaptive CenterNet" from Journal
Paper link:**

# The project upload is not complete, and the full version will be announced when the paper is received.

In recent years, deep learning technologies for object detection have made great progress and have powered the emergence of state-of-the-art models to address object detection problems. Since the domain shift can make detectors unstable or even crash, the detection of cross-domain becomes very important for the design of object detectors. However, traditional deep learning technologies for object detection always rely on a large amount of reliable ground-truth labelling that is laborious, costly, and time-consuming. In this paper, a stepwise domain adaptation detection method (SDA) is proposed for cross-domain object detection. Domain shift is addressed in two steps. In the first step, to bridge the domain gap, an unpaired image-to-image translator is trained to construct a fake target domain by translating the source images to the similar ones in the target domain. In the second step, to further minimize 𝓗-divergence between two domains, an adaptive CenterNet is designed to align distributions at the feature level in an adversarial learning manner. Our proposed method is evaluated in domain shift scenarios based on the driving datasets including Cityscapes, Foggy Cityscapes, SIM10k, and BDD100K. The results show that our method is superior to the state-of-the-art methods and is effective for object detection in domain shift scenarios. 

## Approaches
Domain shift is addressed in two steps. In the first step, to bridge the domain gap, an unpaired image-to-image translator is trained to construct a fake target domain by translating the source images to the similar ones in the target domain. In the second step, an adaptive CenterNet is designed to align distributions at the feature level in an adversarial learning manner.

## Evaluation
Our proposed method is evaluated in two domain shift scenarios based on the driving datasets including Cityscapes, Foggy Cityscapes, and BDD100K. 
### Scenario 1：Clear-to-Haze Adaptation

### Scenario 1：Daytime-to-Nighttime Adaptation


The results show that our method is superior to the state-of-the-art methods and is effective for object detection in domain shift scenarios.
