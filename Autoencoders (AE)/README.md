The objective in this project is to explore Autoencoders and Variational Autoencoders for image reconstruction of diverse private Fotos.

Convolutional Autoencoders are also explored in convolutional_autoencoder.py.

The images needed some preprocessing which was made with data_pre_processing.py. Which also resized and organized the resized fotos in new folders, both in color and black/white.

Finally in autoencoder_generation.py a framework is defined for building an autoencoder based on a configuration file, defining number of hidden layers and neurons in each layers. Also type of layer is specified.
This framework is used for defining an Autoencoder based on a gene and then running a gentic algorithm to find the best possible structure of autoencoder for the given problem. 