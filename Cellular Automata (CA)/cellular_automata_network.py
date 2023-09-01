#%%
import gc
import torch
from torch import nn
import numpy as np
from PIL import Image
from IPython.display import HTML
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from time import sleep
import torch_directml
device = torch_directml.device()

def build_filters(channels_in=3, channels_out=3,kernel_size=9):
    filters = []
    ksize = kernel_size
    total_filters = channels_in*channels_out
    for theta in np.arange(0, np.pi, np.pi / total_filters):
        kern = cv2.getGaborKernel(
            (ksize, ksize), 
            sigma=4.0, #4.0
            theta=theta, 
            lambd=10.0, #10.0
            gamma=0.5, 
            psi=  0.5, 
            ktype=cv2.CV_32F
        )
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return np.array(filters).reshape(
        (channels_in, channels_out, kernel_size, kernel_size)
    )


### Cellular Automata Network
class CellularAutomataNetwork(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        channels =3
        F=9
        kernel_size = (F, F)
        padding = int((F-1)/2)

        # Initialize the conv layer with shape [3,3,3,3] with the following kernel:
        self.conv = nn.Conv2d(
            channels,
            3,
            kernel_size, 
            padding=padding,
            #bias=False
        )
        conv_filters = build_filters(
            channels_in=channels,
            channels_out=3,
            kernel_size=F
        )

        self.conv.weight.data = torch.from_numpy(conv_filters).float().unsqueeze(1)

        self.convt = nn.ConvTranspose2d(
            3,
            channels,
            kernel_size,
        padding=padding,
        #bias=False
        )
        self.relu = nn.ReLU()
        self.sigmoid = nn.Tanh()
        convt_filters = build_filters(
            channels_in=channels,
            channels_out=3,
            kernel_size=F
        )

        self.conv.weight.data = torch.from_numpy(convt_filters).float() #.unsqueeze(1)


    def forward(self, x):
        z = self.conv(x)
        #z = self.relu(z)
        #z = self.convt(z)
        z = self.sigmoid(z)
        return z
    
def tensor_to_image(tensor):
    c=tensor.shape[0]
    w=tensor.shape[1]
    h=tensor.shape[2]
    image = tensor.reshape(w,h,c).detach().to("cpu").numpy()
    image = image*255
    image=image.astype(
        np.uint8
    )
    gc.collect()
    return image



model = CellularAutomataNetwork().to(device)

model.conv.weight.data.shape

#%%
### Initialize random image tensor
tensor = torch.rand(3, 512, 512)
c=tensor.shape[0]
w=tensor.shape[1]
h=tensor.shape[2]
image = tensor_to_image(tensor)
cv2.imshow("start_image", image)
cv2.waitKey()
cv2.destroyAllWindows()
#%%
tensor = tensor.to(device)
tensor = model(tensor)
image = tensor_to_image(tensor)
cv2.imshow("start_image", image)
cv2.waitKey()
cv2.destroyAllWindows()

#%%
from time import sleep

from PIL import Image
import torchvision.transforms as transforms

image = cv2.imread('DSC03540_3.JPG') 
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
rows, cols, _channels = map(int, image.shape)
image = cv2.pyrDown(image, dstsize=(cols // 2, rows // 2))
rows, cols, _channels = map(int, image.shape)
image = cv2.pyrDown(image, dstsize=(cols // 2, rows // 2))

cv2.imshow('Cellular Automata', image)
cv2.waitKey()
cv2.destroyAllWindows()

#%%
# Define a transform to convert PIL 
# image to a Torch tensor
transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Resize((972,1296))
    #transforms.Resize((512,512))
])
  

tensor = transform(image)
#%%
tensor = torch.rand(3, 512, 512).to(device)
import gc 
#idx = torch.randperm(tensor.shape[0])
#tensor = tensor[idx].view(tensor.size())
counter = 0

# TODO: Look for an alternative to ConvTranspose, since the conv2D in pytorch generates the subdivision in 9 parts of the image
# Maybe use a different library for the conv2D
while True:
    tensor = model(tensor)
    img = tensor_to_image(tensor)
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    cv2.imshow('Cellular Automata', img)
    #sleep(1)
    counter += 1
    if counter % 1000 == 0:
        gc.collect()
        print("counter = {0}k, {1}M".format(counter/1000,counter/1e6))

    if cv2.waitKey(1) == ord('q'):
        # press q to terminate the loop
        cv2.destroyAllWindows()
        break

# %%
img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

cv2.imshow('Cellular Automata', img_hsv)
cv2.waitKey(0)
# %%
mantis_img =cv2.imread('mantis4.JPG')
# %%
