# DilatedResNets
Pytorch implementation of https://www.cs.princeton.edu/~funk/drn.pdf


![Alt text](blend103.png "Road segmentation")

# Requirements
Pytorch
Python 3.x
Tensorboad-pytorch
OpenCV
+ Anaconda packages

# Train [For road segmentation]

(a) Download the "fine" cityscapes dataset: https://www.cityscapes-dataset.com/
(b) Create a folder for the dataset with the following structure:

cityscapes
-images
--test
--train
--val
-gt
--test
--train
--val

(c) After setting the right arguments, run main_train.py 
