#%%
#import pygad

from tqdm import tqdm

import pandas as pd
import cv2

import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torch_directml
device = torch_directml.device()

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Resize, Compose, Grayscale
from PIL import Image
from collections import OrderedDict

import plotly.express as px

from statistics import mean

class GeneratedModule(nn.Module):
    def __init__(self, 
            config: OrderedDict,
        ):
        super(GeneratedModule, self).__init__()
        self.layer_config = config["layer_config"]
        self.reshape_config = config["reshape_config"]
        
        self.activations_dict = {
            'ReLU': nn.ReLU(),
            'Sigmoid': nn.Sigmoid(),
            'Tanh': nn.Tanh(),
            "LeakyReLU": nn.LeakyReLU(),
            "ELU": nn.ELU()
        }

        self.layer_types = {
            "Linear": nn.Linear,
            "Convolutional": nn.Conv2d,
            "ConvolutionalTranspose": nn.ConvTranspose2d
        }

        self.pooling_types = {
            "MaxPool": nn.MaxPool2d,
            "AvgPool": nn.AvgPool2d
        }

        self.create_layers_from_config()    

    def create_layers_from_config(self): 
        self.layers_dict=OrderedDict()
        n=0
        layers = list(self.layer_config.keys())
        n_layers = len(layers)-1
        for key, value in self.layer_config.items():
            #print(key, value)
            current_index = layers.index(key)
            current_layer_type = value[0]
            current_layer_neurons = value[1]
            next_layer = layers[current_index+1]
            next_layer_neurons = self.layer_config[next_layer][1]
            layer_name= "Layer_{0}".format(n)
            if current_layer_type in ("Convolutional","ConvolutionalTranspose"):
                kernel_size = value[3]
                pooling=value[4]
                self.__setattr__(
                    layer_name, 
                    self.layer_types[current_layer_type](
                        in_channels=current_layer_neurons, 
                        out_channels=next_layer_neurons, 
                        kernel_size=kernel_size,
                        stride=1, 
                        padding=1,
                ))
            elif current_layer_type == "Linear":
                pooling=None
                self.__setattr__(layer_name, self.layer_types[current_layer_type](current_layer_neurons, next_layer_neurons))

            self.layers_dict["Layer_{0}".format(n)] = (value[2], pooling) # Sets activation function for that layer
            n+=1
            if n==n_layers:
                break

    def forward(self, x):
        try:
            #if self.reshape_config["input"]==1:
            #    z = torch.flatten(x, start_dim=1)
            #else:
            #    
            z=x
            for layer_name in self.layers_dict:
                activation = self.layers_dict[layer_name][0]
                #pooling = self.layers_dict[layer_name][1]
                z = self.activations_dict[activation](self.__getattr__(layer_name)(z))
                #if pooling is not None:
                    #z = self.pooling_types[pooling](z)
            if self.reshape_config["output"] is not None:
                reshape_tuple=self.reshape_config["output"]
                z=z.reshape(reshape_tuple)
            return z
        except Exception as error:
            raise error

class GeneratedAutoencoder(nn.Module):
    def __init__(self, config):
        super(GeneratedAutoencoder, self).__init__()
        self.width = config["width"]
        self.height = config["height"]
        self.channels = config["channels"]
        self.img_type = config["img_type"]
        self.img_subpath=config["img_subpath"]
        #self.input_dim = self.width * self.height * self.channels
        self.modules_config = config["modules"]
        for module, module_config in self.modules_config.items():
            self.__setattr__(module,  GeneratedModule(config=module_config))

        self.get_data_loader()

    def forward(self, x):
        try:
            z=x
            for module in self.modules_config:
                z=self.__getattr__(module)(z)
            return z
        except Exception as error:
            raise error
    
    def get_data_loader(self, sub_path: str=r"\plant\flower"):
        img_folder = ImageFolder(
        root=r"C:\Users\jange\Pictures\all_classified_{0}x{1}_{2}{3}".format(
            self.width,
            self.height,
            self.img_type,
            sub_path
        ), 
        transform=Compose([
            ToTensor(), 
            #Resize(size=(28,28))
        ]
        ),
        loader=self.custom_pil_loader,
        )
        batch_size=5
        self.data = torch.utils.data.DataLoader(img_folder, batch_size=batch_size, shuffle=True)
        return

    def custom_pil_loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
            with open(path, 'rb') as f:
                img = Image.open(f)
                img.load()
                return img
    
    def train(self, epochs=20):
        self.loss_list=[]
        self.avg_loss_list=[]
        self.rolling_avg_loss=[]
        self.avg_loss = 0
        counter=0
        opt = torch.optim.Adam(self.parameters())
        for epoch in tqdm(range(epochs)):
            for x, y in self.data:
                #print(x.shape)
                counter+=1
                x = x.to(device) # GPU
                opt.zero_grad()
                x_hat = self(x)
                loss = ((x - x_hat)**2).sum()
                
                self.loss_list.append(loss.to('cpu').detach().numpy().tolist())
                self.avg_loss = self.avg_loss*(counter-1)/counter+loss.to('cpu').detach().numpy().tolist()/counter # Equation for more efficient incremental mean calculation
                self.rolling_avg_loss.append(mean(self.loss_list[-5:])) 
                self.avg_loss_list.append(self.avg_loss)

                #print("".format(loss))
                loss.backward()
                
                opt.step()
        return


def create_autoencoder_config(
        width,
        height,
        channels,
        encoder_hidden_dim, 
        decoder_hidden_dim: int,
        encoder_input_act_f: str="ReLU",
        encoder_hidden_act_f: str="Sigmoid",
        decoder_latent_dim_act_f: str="ReLU",
        decoder_hidden_dim_act_f: str="Sigmoid",
        latent_dims: int=2,
    ):
    autoencoder_config=OrderedDict()

    autoencoder_config["latent_dims"]=latent_dims
    autoencoder_config["width"]=width
    autoencoder_config["height"]=height
    autoencoder_config["channels"]=channels
    autoencoder_config["img_type"]=img_type
    autoencoder_config["img_subpath"]=r"\plant\flower"

    input_dims = width*height*channels
    output_dims = input_dims

    encoder_config=create_encoder_config(
        input_dim=input_dims, 
        input_act_f=encoder_input_act_f,
        hidden_dim=encoder_hidden_dim,
        hidden_dim_act_f=encoder_hidden_act_f,
        latent_dims=latent_dims
    )

    decoder_config = create_decoder_config(
        channels=channels,
        width=width,
        height=height,
        hidden_dim=decoder_hidden_dim,
        latent_dims=latent_dims,
        latent_dim_act_f=decoder_latent_dim_act_f,
        hidden_dim_act_f=decoder_hidden_dim_act_f,
    )

    ### Create Autoencoder config
    autoencoder_config["modules"]={
        "encoder": encoder_config,
        "decoder": decoder_config
    }
    return autoencoder_config

def create_encoder_config(
        input_dim, 
        input_act_f, 
        hidden_dim, 
        hidden_dim_act_f,
        latent_dims
    ):
    ### Create encoder config
    encoder_layer_config=OrderedDict()
    encoder_layer_config["input_dim"]=(input_dim, "Convolutional",input_act_f, 5, "MaxPool")
    encoder_layer_config["hidden_dim_1"]=(hidden_dim, "Convolutional", hidden_dim_act_f, 3, "MaxPool")
    encoder_layer_config["latent_dim"]=(latent_dims, None)
    encoder_reshape_config = {
        "input": 1,
        "output": None
    }

    encoder_config = {
        "layer_config": encoder_layer_config,
        "reshape_config": encoder_reshape_config
    }
    return encoder_config

def create_decoder_config(
        channels, 
        width, 
        height, 
        hidden_dim: int, 
        latent_dim_act_f: str,
        hidden_dim_act_f: str,
        latent_dims
    ):
    ###  Create decoder config 
    decoder_layer_config=OrderedDict()
    decoder_layer_config["latent_dim"]=(latent_dims, "ConvolutionalTranspose",latent_dim_act_f, 5, "MaxPool")
    decoder_layer_config["hidden_dim_1"]=(hidden_dim, "ConvolutionalTranspose", hidden_dim_act_f,3, "MaxPool")
    decoder_layer_config["output_dim"]=(width*height*channels, None)

    decoder_reshape_config = {
        "input": None,
        #"output": (-1, "channels", "width", "height")
        "output": (-1, channels, width, height)
    }

    decoder_config = {
        "layer_config": decoder_layer_config,
        "reshape_config": decoder_reshape_config
    }
    return decoder_config

#%%
### Test Generating a single autoencoder config file
latent_dims = 2
        
width=30
height=40
img_type="color"

if img_type=="black_white":
    channels=1
elif img_type=="color":
    channels=3
else:
    raise Exception("Img type invalid")



ae_config=create_autoencoder_config(
    width=width,
    height=height,
    channels=channels,
    encoder_hidden_dim=800,
    decoder_hidden_dim=800,
    latent_dims=latent_dims
)
#%%
### Create and Train Autoencoder from config file
autoencoder = GeneratedAutoencoder(config=ae_config).to(device)

autoencoder.train()
px.line(autoencoder.loss_list)
px.line( autoencoder.avg_loss_list)
#%%


latent_dims = 2
        
width=30
height=40
img_type="color"

if img_type=="black_white":
    channels=1
elif img_type=="color":
    channels=3
else:
    raise Exception("Img type invalid")

#%%
### Define fitness function and genetic algorithm
def get_act_f_from_gene(sol):
    if sol < 0.2:
        act_f = 'relu'
    elif sol < 0.4:
        act_f = 'sigmoid'
    elif sol < 0.6:
        act_f = 'tanh'
    elif sol < 0.8:
        act_f = "leaky_relu"
    else:
        act_f = "elu"
    return act_f

def fitness_func(ga_instance, solution, solution_idx):
    min_neurons=2
    max_neurons=2400
    encoder_hidden_dim=int(min_neurons+solution[0]*(max_neurons-min_neurons))
    decoder_hidden_dim=int(min_neurons+solution[1]*(max_neurons-min_neurons))
    encoder_hidden_act_f=get_act_f_from_gene(sol=solution[2])
    encoder_input_act_f=get_act_f_from_gene(sol=solution[3])
    decoder_hidden_dim_act_f=get_act_f_from_gene(sol=solution[4])
    decoder_latent_dim_act_f=get_act_f_from_gene(sol=solution[5])
    ae_config=create_autoencoder_config(
        width=width,
        height=height,
        channels=channels,
        encoder_hidden_dim=encoder_hidden_dim,
        decoder_hidden_dim=decoder_hidden_dim,
        encoder_hidden_act_f=encoder_hidden_act_f,
        encoder_input_act_f=encoder_input_act_f,
        decoder_hidden_dim_act_f=decoder_hidden_dim_act_f,
        decoder_latent_dim_act_f=decoder_latent_dim_act_f,
        latent_dims=latent_dims
    )
    autoencoder = GeneratedAutoencoder(config=ae_config).to(device)
    autoencoder.train()
    fitness = 1.0 / autoencoder.loss_list[-1]
    print("Solution: {0}, fitness: {1}".format(solution, fitness))
    return fitness

# %%
### Run genetic algorithm to find best autoencoder Structure
import pygad

ga_instance = pygad.GA(num_generations=100,
                       num_parents_mating=6,
                       fitness_func=fitness_func,
                       sol_per_pop=10,
                       num_genes=6,
                       init_range_low=0.0,
                       init_range_high=1.0,
                       mutation_percent_genes=0.01,
                       mutation_type=None,
                       gene_type=float,
                       mutation_by_replacement=False,
                       random_mutation_min_val=2,
                       random_mutation_max_val=2400,
                       save_best_solutions=True,
                       save_solutions=False,
                       )
# %%
ga_instance.run()
# %%
### Get best solution from genetic algorithm
best_solution = ga_instance.best_solution()[0]
# array([0.14583265, 0.73472413, 0.65772511, 0.99080862, 0.32838301,0.62063123])
#array([0.08367105, 0.30473803, 0.38084383, 0.04431233, 0.73678686,0.81117514])
# Solution: [0.08367105 0.30473803 0.38084383 0.04431233 0.73678686 0.81117514], fitness: 0.013850880067495987
best_solution = [0.08367105, 0.30473803, 0.38084383, 0.04431233, 0.73678686, 0.81117514]
#%%
### Create and train autoencoder from best solution and try soem ore things out
min_neurons=2
max_neurons=2400
encoder_hidden_dim=int(min_neurons+best_solution[0]*(max_neurons-min_neurons))
decoder_hidden_dim=int(min_neurons+best_solution[1]*(max_neurons-min_neurons))
encoder_hidden_act_f=get_act_f_from_gene(sol=best_solution[2])
encoder_input_act_f=get_act_f_from_gene(sol=best_solution[3])
decoder_hidden_dim_act_f=get_act_f_from_gene(sol=best_solution[4])
decoder_latent_dim_act_f=get_act_f_from_gene(sol=best_solution[5])
ae_config=create_autoencoder_config(
    width=width,
    height=height,
    channels=channels,
    encoder_hidden_dim=encoder_hidden_dim,
    decoder_hidden_dim=decoder_hidden_dim,
    encoder_hidden_act_f=encoder_hidden_act_f,
    encoder_input_act_f=encoder_input_act_f,
    decoder_hidden_dim_act_f=decoder_hidden_dim_act_f,
    decoder_latent_dim_act_f=decoder_latent_dim_act_f,
    latent_dims=latent_dims
)
autoencoder = GeneratedAutoencoder(config=ae_config).to(device)
autoencoder.train()
# %%
### Plot loss and average loss
def plot_latent(autoencoder, data, num_batches: int=None):
    x_list=[]
    y_list=[]
    color_list=[]
    if num_batches is None:
        num_batches = len(data)
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x)
        z = z.to('cpu').detach().numpy()
        x_list.append(z[0][0])
        y_list.append(z[0][1])
        color_list.append(str(y.tolist()[0]))
        if i > num_batches:
            #plt.colorbar()
            break
    df=pd.DataFrame(data={
        "x": x_list,
        "y": y_list,
        "label": color_list
    })
    
    return df #px.scatter(df, x="x", y="y", color="label")

### Plot latent space
def plot_reconstructed(autoencoder, r0=(-4, 4), r1=(-3, 3), n=24):
    w=autoencoder.width
    h=autoencoder.height
    c=autoencoder.channels
    img = np.zeros((n*w, n*h, c))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.reshape(w, h, c).to('cpu').detach().numpy()
            img[(n-1-i)*w:(n-1-i+1)*w, j*h:(j+1)*h, :] = x_hat
    cv2.imshow("recosntructed autoencoder", img)
    cv2.imwrite("test_recosntructed.jpg", img)
    cv2.waitKey()
    return img
# %%
autoencoder=autoencoder.to("cpu")
df =plot_latent(autoencoder=autoencoder, data=autoencoder.data, num_batches=100)
px.scatter(df, x="x", y="y", color="label")
#%%