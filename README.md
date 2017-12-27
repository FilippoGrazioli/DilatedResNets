# DilatedResNets
Pytorch implementation of https://www.cs.princeton.edu/~funk/drn.pdf


![Alt text](blend103.png "Road segmentation")

# Requirements
+ Pytorch
+ Python 3.x
+ Tensorboad-pytorch
+ OpenCV
+ Anaconda packages

# Train [For road segmentation]

(a) Download the "fine" cityscapes dataset: https://www.cityscapes-dataset.com/

(b) Create a folder for the dataset with the following structure:

    ├──
    ├── images                    
    │   ├── test              # Training sample images
    │   ├── train         
    │   └── val 
    ├── gt                    # Ground truths
    │   ├── test         
    │   ├── train        
    │   └── val    
    └── ...
    

(c) After setting the right arguments, run main_train.py 

Note: you can also download pre-trained models, just set --pretrained tag as True. It will download the weights from the Princeton website.

# Inference
TODO: I still have to write a main_inference.py script where you can run a model your trianed on a setof images.
