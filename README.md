# Image-Resolution-Enhancement-using-GANs
The obejective of this project is to take a low resolution images and tries to generate a high resolution images which looks as realistic as possible. This is achieved by using GANs.


## Table of contents
* [General info](#general-info)
* [Project Structure](#project_str)
* [Dataset](#data)
* [Project Execution Steps](#project)
* [Result Similar Images](#similar)
* [Conclusion](#conclusion)
* [Future Work](#future)

<a name="general-info"></a>
## General info

Generative adversarial networks (GANs) have found many applications in Deep Learning. One interesting problem that can be better solved using GANs is to improve the quality of the images. The task objective is upscaling images from low-resolution sizes into high-resolution sizes.Thereby we will use the concept of SRGANs or Super Resolution GANs.

We define the training procedure in the following steps:

- We process the HR(High Resolution) images to get down-sampled LR(Low Resolution) images. Now we have both HR and LR images for training data set.
- We pass LR images through Generator which up-samples and gives SR(Super Resolution) images.
- We use a discriminator to distinguish the HR images and back-propagate the GAN loss to train the discriminator and the generator.


<div style="text-align: center"><img src="data/srgan.jpg" width="700"/></div>
