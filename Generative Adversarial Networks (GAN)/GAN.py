#%%
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import plotly.express as px
from torchvision.transforms import ToTensor, Resize, Compose, Grayscale, functional
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from IPython.display import HTML
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results
import torch_directml
device = torch_directml.device()
#device="cpu"


# Root directory for dataset
dataroot = r"C:\Users\jange\Pictures\all_classified_243x324_color\plant\flower\garden_flower"

# Number of workers for dataloader
workers = 1


# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64





# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 12, 4, 1, 0, bias=True),
            nn.BatchNorm2d(ngf * 12),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 12, ngf * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( ngf * 4, ngf, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=True),
            nn.Sigmoid()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)
    

# Create the generator
netG = Generator(ngpu).to(device)



# Apply the ``weights_init`` function to randomly initialize all weights
#  to ``mean=0``, ``stdev=0.02``.
netG.apply(weights_init)

# Print the model
print(netG)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    
# Create the Discriminator
netD = Discriminator(ngpu).to(device)



# Apply the ``weights_init`` function to randomly initialize all weights
# like this: ``to mean=0, stdev=0.2``.
netD.apply(weights_init)

# Print the model
print(netD)
#%%


# Initialize the ``BCELoss`` function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.
# Batch size during training
batch_size = 128

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr*15, betas=(beta1, 0.999))

# Custom Dataloader
width = 243
height = 324
img_type = "color"
sub_path = r"\plant\flower\garden_flower"

def custom_pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            img.load()
            img = img.convert('HSV')
            return img
        
def get_data_loader(sub_path: str=r"\plant\flower\garden_flower", batch_size: int=10, shuffle: bool=False):
        img_folder = ImageFolder(
        root=r"C:\Users\jange\Pictures\all_classified_{0}x{1}_{2}{3}".format(
            width,
            height,
            img_type,
            sub_path
        ), 
        transform=Compose([
            ToTensor(), 
            #Resize(size=(28,28))
        ]
        ),
        loader=custom_pil_loader,
        )

        data = torch.utils.data.DataLoader(
            img_folder, 
            batch_size=batch_size, 
            shuffle=shuffle,
        )
        return data




dataset = dset.ImageFolder(root=dataroot,
                           transform=#
                           transforms.Compose([
                               transforms.Grayscale(num_output_channels=3),
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
                        )
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True)

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0
# Number of training epochs
num_epochs = 50 # 200

train_discriminator = True
train_generator = True
print("Starting Training Loop...")
# For each epoch
for epoch in tqdm(range(num_epochs)):
    #print("Epoch: {0}".format(epoch))
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        #print(i)
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        if train_discriminator:
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        if train_discriminator:
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        
        if train_generator:
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()            

        # Save Losses for plotting later
        if train_generator:
            G_losses.append(errG.item())
        if train_discriminator:
            D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1
    # Output training stats
    logging.info('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
# %%

px.line(G_losses)
# %%
px.line(D_losses)

# %%
torch.save(netD.state_dict(), "GAN_discriminator_1_st.pt")
torch.save(netD, "GAN_discriminator_1.pt")

torch.save(netG.state_dict(), "GAN_generator_1_st.pt")
torch.save(netG, "GAN_generator_1.pt")
#%%
import cv2
i=0
import torchvision.transforms as T
# define a transform to convert a tensor to PIL image
transform = T.ToPILImage()

# convert the tensor to PIL image using above transform
w=530
h=530  
c=3
np_img_list = [i.reshape(w, h, c).to('cpu').detach().numpy() for i in img_list]
#len(np_img_list)
#%%
img=np_img_list[-1]
#%%
rgbimg = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
cv2.imshow("GAN_result", rgbimg)
cv2.waitKey()
#%%
while True:
    
      
    # divided the image into left and right part
    # like list concatenation we concatenated
    # right and left together
  
    img1 = np_img_list[i]
      
    # this function will concatenate
    # the two matrices
    cv2.imshow('animation', img1)
    i += 1
    if i >= len(g):
        i = 0
    if cv2.waitKey(1) == ord('q'):
        
        # press q to terminate the loop
        cv2.destroyAllWindows()
        break
# %%
