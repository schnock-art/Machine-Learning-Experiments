The objective in this project is to explore GAN for image generation/reconstruction based on fotos or already edited fotos.

The code was adapted from https://github.com/pytorch/tutorials/blob/main/beginner_source/dcgan_faces_tutorial.py

Works quite well, but not so well for colored images, they tend to be ina specific hue.

The complexity of the generator and discriminator should be balanced with the number of neurons per layer (number of channels in the convolutions).

The idea is that the discriminator should be able to discriminate some but not all images, so the generator has a chance to improve. If from the start the loss of the generator is 1, it does not know in which direction to train.


The generator should be generally more complex than the discriminator. Which makes sense if you think about it, discriminating if an image is real or not is easier when you are starting out with noise, while the generator should create something complex from the start out.

It would be interesting to change the architecture of the discriminator over time, first being very simple, and as the loss of the generator decreases, a new more complex discriminator is used.