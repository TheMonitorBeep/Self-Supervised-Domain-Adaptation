# Self Supervised Domain Adaptation using Adversarial Learning
This repository contains the codes for the Supervised Learning Project taken under [Prof. Biplab Banerjee](https://biplab-banerjee.github.io/) and [Prof. Prabhu Ramachandran](https://www.aero.iitb.ac.in/~prabhu/) during Autumn 2020. 

## Problem Statement
The success of machine learning methods on visual recognition tasks is highly dependent on access to large labeled datasets. Unfortunately, performance often drops significantly when the model is presented with data from a new deployment domain which it did not see in training, a problem known as _dataset shift_. Domain Adaptation techniques are used to transfer source knowledge (from the source domain) and adapt it to novel target domains.

## Dataset
For the purpose of this project, I have used two different Domain Adaptation datasets:
* [VisDA 2017 Dataset](http://ai.bu.edu/visda-2017/)
* [Office-Home Dataset](http://hemanthdv.org/OfficeHome-Dataset/)

### VisDA 2017 Dataset
The VisDA dataset contains two different domains (for training), the training and the validation domain with images from 12 different object categories in both the domains. Images in the training domain are synthetic 2D renderings of 3D models generated from different angles and with different lighting conditions, making them easy to label with minimal human intervention. Images in the validation domain are real life images with a varied underlying distribution which makes them quite different from the source domain images. The model is trained on the synthetic images (source-domain) and then validated on the real images (target-domain). The model can then also be tested on a test domain (not done here). 
> Note: The validation domain images provided in the dataset are labelled, however, the problem statement tries to address scenarios where labelled datasets in the target domain are unavailable. Hence, for this project, the model is trained considering the real images in the validation domain as unlabelled. The dataset can be downloaded from the above link. 

![](http://ai.bu.edu/visda-2017/assets/images/classification-shift.png)

### Office-Home
The Office-Home dataset is another dataset used for the evaluation of domain adaptation techniques. As compared to the VisDA dataset, Office-Home contains 4 different domains, namely, Artistic images, Clip Art, Product images and Real-World images. For each domain, the dataset contains images of 65 object categories found typically in Office and Home settings. There are a total of 15500 images distributed among 65 different object categories. The dataset can be downloaded from the above link.

![](http://hemanthdv.github.io/profile/images/DataCollage.jpg)

## Domain Adversarial Neural Network (DANN)
This is an unoffical implementation of the Domain Adversarial Neural Network mentioned in this paper. 
