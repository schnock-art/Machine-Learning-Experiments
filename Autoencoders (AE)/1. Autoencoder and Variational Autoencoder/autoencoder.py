#%%
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions


import torchvision
import numpy as np
import matplotlib.pyplot as plt#; plt.rcParams['figure.dpi'] = 200
from tqdm import tqdm
import torch_directml
import pandas as pd
import cv2
device = torch_directml.device()

class Encoder(nn.Module):
    def __init__(self, 
                 latent_dims, 
                 input_dim, 
        ):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.linear1 = nn.Linear(self.input_dim, 1000)
        self.linear2 = nn.Linear(1000, 512)
        self.linear3 = nn.Linear(512, latent_dims)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        x = F.sigmoid(self.linear2(x))
        return self.linear3(x)
    
class Decoder(nn.Module):
    def __init__(self, latent_dims, width, height, channels):
        super(Decoder, self).__init__()
        self.width = width
        self.height = height
        self.channels = channels
        self.output_dim = self.width * self.height * self.channels
        self.linear1 = nn.Linear(latent_dims, 1000)
        self.linear2 = nn.Linear(1000, 512)
        self.linear3 = nn.Linear(512, self.output_dim)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.relu(self.linear2(z))
        z = torch.sigmoid(self.linear3(z))
        return z.reshape((-1, self.channels, self.width, self.height))
    
class Autoencoder(nn.Module):
    def __init__(self, latent_dims, width, height, channels):
        super(Autoencoder, self).__init__()
        self.width = width
        self.height = height
        self.channels = channels
        self.input_dim = self.width * self.height * self.channels
        self.encoder = Encoder(
            latent_dims=latent_dims, 
            input_dim=self.input_dim
        )
        self.decoder = Decoder(
            latent_dims=latent_dims, 
            width=width, 
            height=height, 
            channels=channels
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    

def train_ae(autoencoder, data, epochs=20):
    loss_list=[]
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in tqdm(range(epochs)):
        for x, y in data:
            x = x.to(device) # GPU
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum()
            loss_list.append(loss.to('cpu').detach().numpy().tolist())
            #print("".format(loss))
            loss.backward()
            opt.step()
    return autoencoder, loss_list


class VariationalEncoder(nn.Module):
    def __init__(self, 
                 latent_dims, 
                 input_dim):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)

        self.N = torch.distributions.Normal(
            torch.tensor(0.0).to(device), 
            torch.tensor(1.0).to(device)
        )
  
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z
    
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, width, height, channels):
        super(VariationalAutoencoder, self).__init__()
        self.width = width
        self.height = height
        self.channels = channels
        self.input_dim = self.width * self.height * self.channels
        self.encoder = VariationalEncoder(
            latent_dims=latent_dims, 
            input_dim=self.input_dim
        )
        self.decoder = Decoder(
            latent_dims=latent_dims, 
            width=width, 
            height=height, 
            channels=channels
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def train_vae(autoencoder, data, epochs=20):
    loss_list=[]
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in tqdm(range(epochs)):
        for x, y in data:
            x = x.to(device) # GPU
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl
            loss_list.append(loss.to('cpu').detach().numpy().tolist())
            #print("".format(loss))
            loss.backward()
            opt.step()
    return autoencoder, loss_list

import plotly.express as px
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

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Resize, Compose, Grayscale
from PIL import Image

def custom_pil_loader(path):
# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        img.load()
        return img

width=30
height=40
img_type="color"
if img_type=="black_white":
    channels=1
elif img_type=="color":
    channels=3
else:
    raise Exception("Img type invalid")

img_folder = ImageFolder(
    root=r"C:\Users\jange\Pictures\all_classified_{0}x{1}_{2}\plant\flower".format(
        width,
        height,
        img_type
    ), 
    transform=Compose([
        ToTensor(), 
        #Resize(size=(28,28))
    ]
    ),
    loader=custom_pil_loader,
)
batch_size=5
data = torch.utils.data.DataLoader(img_folder, batch_size=batch_size, shuffle=True)
#%%
#%%
#### TRAINING  ########################
latent_dims = 2
# width = 30
# height = 40#
#channels=1
autoencoder = Autoencoder(
    latent_dims=latent_dims,
    width=width,
    height=height,
    channels=channels
).to(device) # GPU

autoencoder, loss = train_ae(autoencoder, data, epochs=50)

pd.Series(loss).to_pickle("loss.pkl")
torch.save(autoencoder, "autoencoder.pt")

#############################
#%%
########## LOAD MODEL #################
import pandas as pd
import torch
path="autoencoder.pt"
model=torch.load(path) #.to("cpu")
model = model.to("cpu")
loss=pd.read_pickle("loss.pkl")
# %%
# Plot loss
px.line(loss)
#%%
# Plot latent space
df = plot_latent(autoencoder=autoencoder.to("cpu"), data=data, num_batches=1000)
px.scatter(df, x="x", y="y", color="label").show()

# %%
# Plot reconstructed images
img = plot_reconstructed(autoencoder, 
                         r0=(-7, 4), 
                         r1=(-5, 5), 
                         n=12)

# %%
### Variational Autoencoder
v_autoencoder=VariationalAutoencoder(
    latent_dims=2,
    width=width,
    height=height,
    channels=channels
).to(device)

v_autoencoder, vae_loss=train_vae(autoencoder=v_autoencoder, data=data, epochs=50)
torch.save(autoencoder, "variational_autoencoder.pt")

pd.Series(vae_loss).to_pickle("vae_loss.pkl")

#%%
# Plot loss
px.line(vae_loss)
#%%
# Plot latent space
v_autoencoder = v_autoencoder.to("cpu")
v_autoencoder.encoder.N = torch.distributions.Normal(
            torch.tensor(0.0).to("cpu"), 
            torch.tensor(1.0).to("cpu")
)
df = plot_latent(autoencoder=v_autoencoder, data=data, num_batches=1000)
px.scatter(df, x="x", y="y", color="label").show()

# %%
# Plot reconstructed images
img = plot_reconstructed(v_autoencoder)
