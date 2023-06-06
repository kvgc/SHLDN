## Semisupervised Heuristic Lens Detector Nets

Source code for MNRAS paper: Optimizing machine learning methods to discover strong gravitational
lenses in the Deep Lens Survey

![pipeline](https://user-images.githubusercontent.com/37102188/198930167-1d22c5ec-a028-461e-be40-29d9c3453a7e.png)

Source code for AISTATS 2022 paper: [An Unsupervised Hunt for Gravitational Lenses](https://arxiv.org/abs/2210.11681)

*Strong gravitational lenses allow us to peer into the farthest reaches of space by bending the light from a background object around a massive object in the foreground. Unfortunately, these lenses are extremely rare, and manually finding them in astronomy surveys is difficult and time-consuming. We are thus tasked with finding them in an automated fashion with few if any, known lenses to form positive samples. To assist us with train- ing, we can simulate realistic lenses within our survey images to form positive samples. Naively training a ResNet model with these simulated lenses results in a poor precision for the desired high recall, because the simulations contain artifacts that are learned by the model. In this work, we develop a lens detection method that combines simulation, data augmentation, semi-supervised learning, and GANs to improve this performance by an order of magnitude. We perform ablation studies and examine how performance scales with the number of non-lenses and simulated lenses. These findings allow researchers to go into a survey mostly “blind” and still classify strong gravitational lenses with high precision and recall.*


## Google drive directory structure:

Link: [Google Drive](https://drive.google.com/drive/folders/1mqEVBVmz3XBGTjibrk6Qu0MRpkOPSbG0?usp=drive_link)

#### Datasets
The original DLS datasets our training data and testing data are made from can be found in the "data/DLS/" directory. The directory contains the ".png" files and the original ".fits" cutout files. Similarly, the training datasets can be found in the "data/TrainingDatasets/" directory and the testing datasets can be found in "data/TestDatasets/" directory.

#### Trained models
All of our trained models are stored in the models directory. The learning method, training set and augmentations used during training are specified in the directory name.
For example, "results/models/pi_model_TrainingV2_with_augmentation_and_GAN" uses the semi-supervised learning method Pi-model along with augmentations and GANs.
Under each model directory, there are 4 versions corresponding to 4 independent runs.

#### Prediction scores for Deep Lens Survey images
We pass all the images from the Deep Lens Survey through all the trained models (i.e. version 4) and store the prediction scores in the results/csv/ folder corresponding to each model. For example, "results/csv/pi_model_TrainingV2_with_augmentation_and_GAN/version 1.csv" contains the prediction scores obtained for the v1 version of the pi-model with augmentaion and GANs
The csv contains the survey objid, along with the prediction score.
