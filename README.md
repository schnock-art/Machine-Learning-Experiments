# Machine-Learning-Experiments
Exploring some Machine Learning Algorithms for image manipulation, creation and reconstruction

## Autoencoders

The objective in this project is to explore Autoencoders and Variational Autoencoders for image reconstruction of diverse private Fotos.

Convolutional Autoencoders are also explored in convolutional_autoencoder.py.

The images needed some preprocessing which was made with data_pre_processing.py. Which also resized and organized the resized fotos in new folders, both in color and black/white.

Finally in autoencoder_generation.py a framework is defined for building an autoencoder based on a configuration file, defining number of hidden layers and neurons in each layers. Also type of layer is specified.
This framework is used for defining an Autoencoder based on a gene and then running a gentic algorithm to find the best possible structure of autoencoder for the given problem. 

## Cellular Automata
The objective is to explore Cellular Automata for image generation. 

## Generative Adversarial Networks
The objective is to explore GAN for image generation/reconstruction based on fotos or already edited fotos.

The code was adapted from https://github.com/pytorch/tutorials/blob/main/beginner_source/dcgan_faces_tutorial.py

Works quite well, but not so well for colored images, they tend to be ina specific hue.