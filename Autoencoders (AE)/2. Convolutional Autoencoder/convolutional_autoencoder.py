
#%%
from tqdm import tqdm

import pandas as pd
import cv2

import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from torch.nn.utils import clip_grad_norm_
import torch.distributions
import torch_directml
device = torch_directml.device()
torch.set_default_dtype(torch.float32)
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Resize, Compose, Grayscale, functional
from PIL import Image
from collections import OrderedDict

import plotly.express as px
import numpy as np
from statistics import mean
import logging
import math

logging.basicConfig(level=logging.INFO)

class ConvolutionalEncoder(nn.Module):
    def __init__(self, 
                input_dim,
                latent_dims, 
                width=30,
                height=40,
                channels=3,
        ):
        super(ConvolutionalEncoder, self).__init__()
        self.input_dim = input_dim
        self.width = width
        self.height = height
        self.channels = channels
        self.output_dim = latent_dims
        self.common_denominator = math.gcd(self.width,self.height)

        self.conv2d_1_out_channels = 3
        self.conv2d_1_kernel_size = 3
        self.conv2d_1 = nn.Conv2d(
            self.channels, 
            self.conv2d_1_out_channels, 
            self.conv2d_1_kernel_size, 
            padding=1,
            #stride=2,
            groups=self.channels,
        )

        self.pool_x_1 = self.width//self.common_denominator
        self.pool_y_1 = self.height//self.common_denominator
        self.pool_1 = nn.MaxPool2d(
            kernel_size=(self.pool_x_1, self.pool_y_1),
        )  

        self.batch_norm_1 = nn.BatchNorm2d(self.conv2d_1_out_channels)

        self.conv2d_2_out_channels = 3
        self.conv2d_2_kernel_size = 3
        self.conv2d_2 = nn.Conv2d(
            self.conv2d_1_out_channels, 
            self.conv2d_2_out_channels,
            self.conv2d_2_kernel_size, 
            #stride=2,
            #groups=self.conv2d_1_out_channels,
            padding=1,
        )
        
        self.pool_x_2 = 2
        self.pool_y_2 = 2
        self.pool_2 = nn.MaxPool2d(self.pool_x_2, self.pool_y_2)
        self.linear1_in_features = (self.width//self.pool_x_2 //self.pool_x_1
            ) * (
             self.height//self.pool_y_2 //self.pool_y_1
             ) * self.conv2d_2_out_channels 
        
        self.batch_norm_2 = nn.BatchNorm2d(self.conv2d_2_out_channels)

        self.linear2_in_features = self.linear1_in_features//4

        self.linear1 = nn.Linear(
            self.linear1_in_features, 
            self.linear2_in_features,
            )
        
        
        
        self.linear2 = nn.Linear(
            self.linear2_in_features, 
            self.output_dim,
        )

        self.linear3 = nn.Linear(
            self.linear2_in_features, 
            self.output_dim,
        )

        self.N = torch.distributions.Normal(
            torch.tensor(0.0).to(device), 
            torch.tensor(1.0).to(device)
        )
  
        self.kl = 0
        

    def forward(self, x):
        logging.debug("Encoder forward")
        #x = torch.flatten(x, start_dim=1)
        logging.debug("Input shape: {0}".format(x.shape))
        x = F.relu(self.conv2d_1(x))
        logging.debug("Conv2d_1 relu shape: {0}".format(x.shape))
        
        x = self.batch_norm_1(x)
        x = self.pool_1(x)
        logging.debug("Pool 1 shape: {0}".format(x.shape))
        x = F.relu(self.conv2d_2(x))
        logging.debug("Conv2d_2 relu shape: {0}".format(x.shape))
        self.batch_norm_2(x)
        x = self.pool_2(x)
        logging.debug("Pool 2 shape: {0}".format(x.shape))
        x = torch.flatten(x, start_dim=1)
        logging.debug("Flatten shape: {0}".format(x.shape))
        x = F.relu(self.linear1(x))
        logging.debug("Linear 1 shape: {0}".format(x.shape))

        #Variational
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()

        logging.debug("Linear 2 shape: {0}".format(x.shape))
        return z
    
    
class ConvolutionalDecoder(nn.Module):
    def __init__(self, 
                latent_dims, 
                width, 
                height, 
                channels
        ):
        super(ConvolutionalDecoder, self).__init__()
        self.width = width
        self.height = height
        self.channels = channels
        self.output_dim = self.width * self.height * self.channels
        
        self.common_denominator = math.gcd(self.width,self.height)
        
        self.conv_transpose_2d_1_in_channels = 8 #self.channels
        self.pool_x_1 = self.width//self.common_denominator
        self.pool_y_1 = self.height//self.common_denominator
        self.pool_x_2 = 2
        self.pool_y_2 = 2
        
        self.linear2_out_features = (
             self.width #//self.pool_x_1//self.pool_x_2
             ) * (
             self.height #//self.pool_y_1//self.pool_y_2
             ) * self.conv_transpose_2d_1_in_channels
        
        self.linear1_out_features = self.linear2_out_features//8

        self.linear1 = nn.Linear(
            latent_dims, 
            self.linear1_out_features,
        )

        self.linear2 = nn.Linear(
            self.linear1_out_features, 
            self.linear2_out_features,
        )

        #self.conv_transpose_2d_1_in_channels = 4
        self.conv_transpose_2d_1_out_channels = 8
        self.conv_transpose_2d_1 = nn.ConvTranspose2d(
             self.conv_transpose_2d_1_in_channels, 
             self.conv_transpose_2d_1_out_channels, 
             kernel_size=3,
             stride=2,
             #stride= self.pool_x_2 #(self.pool_x_2, self.pool_y_2),
             padding=1,
            )
        
        self.batch_norm_1 = nn.BatchNorm2d(self.conv_transpose_2d_1_out_channels)

        self.conv_transpose_2d_2_in_channels = self.conv_transpose_2d_1_out_channels
        self.conv_transpose_2d_2 = nn.ConvTranspose2d(
            self.conv_transpose_2d_2_in_channels, 
            self.channels,
            kernel_size=3,
            #stride=2,
            #groups=3,
            #stride=(self.pool_x_1, self.pool_y_1),
            padding=1,
        )
        self.batch_norm_2 = nn.BatchNorm2d(self.channels)

    def forward(self, z):
        logging.debug("Decoder forward")
        logging.debug("Input shape: {0}".format(z.shape))
        z = F.relu(self.linear1(z))
        logging.debug("Linear 1 relu shape: {0}".format(z.shape))
        z = F.sigmoid(self.linear2(z))
        logging.debug("Linear 2 sigmoid shape: {0}".format(z.shape))
        z = z.reshape((-1, 
                       self.conv_transpose_2d_1_in_channels, 
                       self.width,#//self.pool_x_2//self.pool_x_1, 
                       self.height, #//self.pool_y_2//self.pool_y_1
                       )
                       )
        logging.debug("Reshape shape: {0}".format(z.shape))
        z = torch.sigmoid(self.conv_transpose_2d_1(z))
        z = self.batch_norm_1(z)
        logging.debug("Conv2d_1 relu shape: {0}".format(z.shape))

        #output_size_1=(self.width//self.pool_x_1, self.height//self.pool_y_1)
        #z = F.interpolate(z, size=output_size_1, mode='nearest')
        #logging.debug("Upsample 1 shape: {0}".format(z.shape))

        output_size_2=(self.width, self.height)
        z = F.interpolate(z, size=output_size_2, mode='nearest')

        logging.debug("Upsample 2 shape: {0}".format(z.shape))
        z = torch.sigmoid(self.conv_transpose_2d_2(z))
        #z = self.batch_norm_2(z)
        logging.debug("Conv2d_2 sigmoid shape: {0}".format(z.shape))
        return z


class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, config):
        super(ConvolutionalAutoencoder, self).__init__()
        self.width = config["width"]
        self.height = config["height"]
        self.channels = config["channels"]
        self.img_type = config["img_type"]
        self.img_subpath = config["img_subpath"]
        self.latent_dims = config["latent_dims"]
        self.input_dim = self.width * self.height * self.channels
        #self.modules_config = config["modules"]
        # for module, module_config in self.modules_config.items():
        #     self.__setattr__(module,  GeneratedModule(config=module_config))
        self.encoder = ConvolutionalEncoder(
            input_dim=self.input_dim,
            latent_dims=self.latent_dims, 
            width=self.width,
            height=self.height,
            channels=self.channels,
        )
        self.decoder = ConvolutionalDecoder(
            latent_dims=self.latent_dims, 
            width=self.width,
            height=self.height,
            channels=self.channels,
        )
        self.latent_space_domain = {
            "x": (-1,1),
            "y": (-1,1),
        }

    def forward(self, x):
        z = self.decoder(self.encoder(x))
        return z

    
    def get_data_loader(self, sub_path: str=r"\plant\flower\garden_flower", batch_size: int=10, shuffle: bool=False):
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

        self.data = torch.utils.data.DataLoader(
            img_folder, 
            batch_size=batch_size, 
            shuffle=shuffle,
        )
        return

    def custom_pil_loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
            with open(path, 'rb') as f:
                img = Image.open(f)
                img.load()
                img = img.convert('HSV')
                return img
    
    def train(self, 
              epochs: int=20,
              sub_path=r"\plant\flower\garden_flower", 
              batch_size: int=10, 
              shuffle: bool=False,
              learning_rate: float=0.0001,
              weight_decay=1e-05,
              betas=(0.9, 0.95),
        ):
        self.loss_list=[]
        self.avg_loss_list=[]
        self.rolling_avg_loss=[]
        self.avg_loss = 0
        counter=0
        epoch_counter=0
        opt = torch.optim.Adam(
            self.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            amsgrad=True,
        )
        batch_size = batch_size
        for epoch in tqdm(range(epochs)):
            #print(epoch)
            
            self.get_data_loader(
            sub_path=sub_path,
            batch_size=batch_size, 
            shuffle=shuffle
            )
            epoch_loss_list = []
            for x, y in self.data:
                counter+=1
                x = x.to(device) # GPU
                opt.zero_grad()
                x_hat = self.forward(x)
                if self.channels==3:
                    gc_transform = Grayscale()
                    x_hat = gc_transform(x_hat)
                logging.debug("x_hat shape: {0}".format(x_hat.shape))
                logging.debug("x: {0}".format(x))
                #logging.debug("x shape: {0}".format(x.shape))
                logging.debug("x_hat: {0}".format(x_hat))
                loss = (((x - x_hat)**2).sum()+ self.encoder.kl)/batch_size/self.width/self.height #/self.channels
                logging.debug(loss)
                loss_value = loss.to('cpu').detach().numpy().tolist()
                logging.debug("Loss: {0}".format(loss_value))
                if math.isnan(loss_value):
                    logging.info("Loss is nan!")
                    return
                self.loss_list.append(loss_value)
                self.avg_loss = self.avg_loss*(counter-1)/counter+loss.to('cpu').detach().numpy().tolist()/counter # Equation for more efficient incremental mean calculation
                self.rolling_avg_loss.append(mean(self.loss_list[-5:]))
                self.avg_loss_list.append(self.avg_loss)
                epoch_loss_list.append(loss_value)
                #logging.debug("".format(loss))
                
                loss.backward()
                #clip_grad_norm_(self.parameters(), args.clip)
                opt.step()
            epoch_loss = mean(epoch_loss_list)
            epoch_var = np.var(epoch_loss_list)
            logging.info("Epoch: {0}, Batch Size: {1} Avg Loss: {2}, Loss variance: {3}".format(epoch, batch_size, epoch_loss, epoch_var))
            epoch_counter+=1
            #if epoch_counter > 1000 and batch_size > 1:
            #    batch_size = batch_size//2
            #    learning_rate = learning_rate/2
            #    opt = torch.optim.Adam(
            #        self.parameters(), 
            #        lr=learning_rate,
            #        weight_decay=weight_decay
            #    )
            #    if batch_size < 1:
            #        batch_size = 1
            #    epoch_counter=0
            ##break
        return
    
    def plot_latent(self, num_batches: int=None):
        x_list=[]
        y_list=[]
        color_list=[]
        num_batches=1000
        i=0
        for x, y in self.data:
            #print(x.shape)
            z = x.to(device)
            z = conv_autoencoder.encoder(z)
            z = z.to('cpu').detach().numpy()
            x_list+=list(z[:,0])
            y_list+=list(z[:,1])
            color_list+=[str(label) for label in y]
            i+=1
            if i > num_batches:
                #plt.colorbar()
                break
        self.latent_space_df=pd.DataFrame(data={
            "x": x_list,
            "y": y_list,
            "label": color_list
        })
        self.latent_space_domain = {
            "x": (min(x_list), max(x_list)),
            "y": (min(y_list), max(y_list)),
        }
        return self.latent_space_df #px.scatter(df, x="x", y="y", color="label")

    def plot_reconstructed(self, 
            r0=None, 
            r1=None, 
            n=24
        ):
        w=self.width
        h=self.height
        c=self.channels
        if r0 is None:
            r0 = self.latent_space_domain["x"]
        if r1 is None:
            r1 = self.latent_space_domain["y"]
        img = np.zeros((n*w, n*h, c))
        for i, y in enumerate(np.linspace(*r1, n)):
            for j, x in enumerate(np.linspace(*r0, n)):
                z = torch.Tensor([[x, y]]).to(device)
                x_hat = self.decoder(z)
                x_hat = x_hat.reshape(w, h, c).to('cpu').detach().numpy()
                rgbimg = cv2.cvtColor(x_hat, cv2.COLOR_HSV2RGB)
                img[(n-1-i)*w:(n-1-i+1)*w, j*h:(j+1)*h, :] = rgbimg
        #img_pil=functional.to_pil_image(img, mode=None)
        cv2.imshow("recosntructed autoencoder", img)
        cv2.imwrite("test_recosntructed.jpg", img)
        cv2.waitKey()
        return img

#%%
autoencoder_config = {
    "width": 30,
    "height": 40,
    "channels": 3,
    "img_type": "black_white",
    "img_subpath": r"\plant\flower\garden_flower",
    "latent_dims": 2,
}

conv_autoencoder = ConvolutionalAutoencoder(
    config=autoencoder_config
).to(device)
#%%

x=conv_autoencoder.train(
    sub_path=r"\plant\flower\garden_flower",
    epochs=100,
    batch_size=100,
    shuffle=True,
    learning_rate=0.00001,
    weight_decay=1e-05,
    betas=(0.9, 0.999),
)
#%%
# Plot Loss
px.line(conv_autoencoder.loss_list)

#%%
# Plot latent space
df = conv_autoencoder.plot_latent()
px.scatter(df, 'x', 'y', color='label')

#%%
# Plot reconstructed images
conv_autoencoder.plot_reconstructed()

# %%
