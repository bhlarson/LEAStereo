# https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        # Define trainable layers in __init__
        super(Net, self).__init__()

        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        self.conv1 = nn.Conv2d(1,32,3,1)
        self.conv2 = nn.Conv2d(32,64,3,1)

        # torch.nn.Dropout2d(p=0.5, inplace=False)
        # https://pytorch.org/docs/stable/generated/torch.nn.Dropout2d.html
        # Dropout as layer rather than functional so it is present in training and absent in inference
        # https://discuss.pytorch.org/t/how-to-choose-between-torch-nn-functional-and-torch-nn-module/2800/13
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        # torch.nn.Linear(in_features, out_features, bias=True)
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # connect layers in forward
        x = self.conv1(x)
       
        # Using functional form for non-trainable layers
        x = F.relu(x)  # RELU activation function over x, https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU
        x = self.conv2(x)
        x = F.relu(x)

        x = F.max_pool2d(x,2) # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d
        x = self.dropout1(x)
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output


# Test network:
# Equates to one random 28x28 image
random_data = torch.rand((1, 1, 28, 28))

my_nn = Net()
result = my_nn(random_data)
print (result)