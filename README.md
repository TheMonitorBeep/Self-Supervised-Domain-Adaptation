# Self Supervised Domain Adaptation using Adversarial Learning
This repository contains the codes for the Supervised Learning Project taken under Prof. Biplab Banerjee and Prof. Prabhu Ramachandran during Autumn 2020. 

## Dataset
For the purpose of this project, I have used two different Domain Adaptation datasets:
* ViSDA 2017 Dataset
* Office-Home Dataset

The VisDA dataset has 12 classes in the source and the target domain. The challenge is to perform domain adaptation between the two domains and report classification accuracy on the test dataset when the model is solely trained on the source domain images. The source domain images are computer generated and thus easy to classify and label in different classes. On the other hand, the target domain images are real life images with a varied underlying distribution which makes them different from the source domain images. 

## Domain Adversarial Neural Network (DANN)
This is an unoffical implementation of the Domain Adversarial Neural Network mentioned in this paper. 
