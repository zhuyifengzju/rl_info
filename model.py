
import numpy as np

"""# Load MNIST Dataset"""

from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader

dataset = CIFAR10('./data', transform=transforms.Compose([#transforms.Resize(size=(256,256)),
                                                        transforms.ToTensor(),
                                                       ]), download=True)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

import torch
"""
Optional: Your code here
"""

import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='parsing')
    parser.add_argument('--load', action='store_true')

    return parser.parse_args()

class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self._nn = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 6, stride=2, padding=1),
            torch.nn.MaxPool2d((2, 2), stride=(2,2)),
            torch.nn.ReLU(True),
             torch.nn.Conv2d(32, 64, 5,stride=2, padding=1),
            torch.nn.MaxPool2d((2, 2), stride=(2,2)),
            torch.nn.ReLU(True),
             torch.nn.Conv2d(64, 128, 5,stride=2, padding=1),
            torch.nn.ReLU(True),
             torch.nn.Conv2d(128, 2048, 5,stride=2, padding=1),
            torch.nn.MaxPool2d((2, 2), stride=(2,2)) 
        )

        self._mu_layer = torch.nn.Linear(400, 100)
        self._var_layer = torch.nn.Linear(400, 100)
        
    def forward(self, img):
        h = self._nn(img)
        h = torch.flatten(h)
        return  h # self._mu_layer(h), self._var_layer(h)
        
class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self._nn = torch.nn.Sequential(
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            torch.nn.ConvTranspose2d(2048, 128, 5, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(128, 64, 5, stride=2),  
            torch.nn.ReLU(True),
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            torch.nn.ConvTranspose2d(64, 32, 5, stride=2),  
            torch.nn.ReLU(True),
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            torch.nn.ConvTranspose2d(32, 3, 6, stride=2),
            torch.nn.Sigmoid(),
        )

    # This function is for VAE
    def reparametrization(self, mu, log_var):
        variance = log_var.mul(0.5).exp()
        if torch.cuda.is_available:
          eps = torch.cuda.FloatTensor(variance.size()).normal_()
        else:
          eps = torch.FloatTensor(variance.size()).normal_()
        eps = torch.autograd.Variable(eps).cuda()
        return eps.mul(variance).add_(mu)

    def forward(self, z):
      # TODO: reshape the tensor from vector to image  
      z = z.view(-1, 2048, 1, 1)
      y = self._nn(z)
      return y

encode = Encoder().cuda()
decode = Decoder().cuda()    

l1_loss = torch.nn.L1Loss(size_average=False)
mse_loss = torch.nn.MSELoss(size_average=False)

param = list(encode.parameters()) + list(decode.parameters())
optimizer = torch.optim.Adam(param, lr=lr,weight_decay=1e-5)

torch.autograd.set_detect_anomaly(True)

def encoder_loss(mu, log_var):
    return torch.sum(log_var.exp() + mu.pow(2) - log_var).mul_(0.5)


# loss = mse_loss(Y, X) + encoder_loss(mu, log_var)
