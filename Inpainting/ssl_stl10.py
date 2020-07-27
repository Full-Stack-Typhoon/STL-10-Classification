import numpy as np 
import os
import os.path
import numpy as np
import cv2
import scipy.io
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib.image as imagelib
from tqdm import tqdm
import random
import time
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import trange
from time import sleep
from torchvision import datasets, models, transforms
import copy
import zipfile
import torchvision.transforms as transforms
import PIL.Image as Image
from torch.autograd import Variable
import torchvision.datasets as datasets
import torch.nn.init as init


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

use_gpu = False
if torch.cuda.is_available():
    device = 'cuda'
    use_gpu = True
else:
    device = 'cpu'

width_in = 568                                                                                            
height_in = 568                                                                                            
width_out = 566                                                                                            
height_out = 566                                                                                            
batch_size = 16
epochs = 1

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

data_transforms = transforms.Compose([
    transforms.Resize((568,568)),
    transforms.ToTensor()
])


train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder('data/train_images/',
                         transform=data_transforms),
    batch_size=16, shuffle=True, num_workers=1)

dataloaders_dict = {'train' : train_loader}


def weight_init(m):
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

train_images = os.listdir('Data_Unlabeled')


class STL10DataSet(Dataset):
    def __init__(self, input_images):
        self.images = input_images
        self.num_image = len(self.images)
    
    def __getitem__(self,index):
        img = self.images[index]
        data = data_transforms(pil_loader("Data_Unlabeled/" + img))
        data = data.view(data.size(0), data.size(1), data.size(2))
        data = Variable(data, volatile=True)
        return data, data
    def __len__(self):
        return self.num_image

class STL10DataSetTest(Dataset):
    def __init__(self, input_images):
        self.images = input_images
        self.num_image = len(self.images)
    
    def __getitem__(self,index):
        img = self.images[index]
        data = data_transforms(pil_loader("data/train_images/0/" + img))
        data = data.view(data.size(0), data.size(1), data.size(2))
        data = Variable(data, volatile=True)
        return data, data
    def __len__(self):
        return self.num_image


class STL10RandomDropout(Dataset):
    def __init__(self, input_images, dropout=0.5):
        self.images = input_images
        self.num_image = len(self.images)
        self.dropout = nn.Dropout(p=dropout)
        self.to_tensor = transforms.ToTensor()
    
    def __getitem__(self,index):
        img = self.images[index]
        data = data_transforms(pil_loader("Data_Unlabeled/" + img))
        data = data.view(data.size(0), data.size(1), data.size(2))
        input_data = self.dropout(data)
        return input_data, data
    
    def __len__(self):
        return self.num_image


class ModSequential(nn.Sequential):
    def __init__(self, *args):
        super(ModSequential, self).__init__(*args)
        
    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input


class Encoder_Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder_Decoder, self).__init__()
        self.l_relu = nn.LeakyReLU(0.2, inplace=True)
        self.tanh = nn.Tanh()
        self.conv2d_1= nn.Conv2d(in_channels, 144, 5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(144)
        self.conv2d_2 = nn.Conv2d(144, 144, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(144)
        self.conv2d_3 = nn.Conv2d(144, 256, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv2d_4 = nn.Conv2d(256,256, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.deconv2d_4 = nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1)
        self.d_bn4 = nn.BatchNorm2d(256)
        self.deconv2d_3 = nn.ConvTranspose2d(256, 144, 3, stride=2, padding=1, output_padding=1)
        self.d_bn3 = nn.BatchNorm2d(144)
        self.deconv2d_2 = nn.ConvTranspose2d(144, 144, 3, stride=2, padding=1, output_padding=1)
        self.d_bn2 = nn.BatchNorm2d(144)
        self.deconv2d_1 = nn.ConvTranspose2d(144, out_channels, 5, stride=2, padding=2, output_padding=1)
        self.dropout = nn.Dropout(p=0.70)
        
        
        self.encoder = ModSequential(
            nn.Conv2d(in_channels, 128, 5, stride=2, padding=2),  nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 144, 3, stride=1, padding=1),  nn.LeakyReLU(0.2, True),
            nn.Conv2d(144, 144, 3, stride=1, padding=1),  nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(144),
            nn.Conv2d(144, 144, 3, stride=2, padding=1),  nn.LeakyReLU(0.2, True),
            nn.Conv2d(144, 144, 3, stride=1, padding=1),  nn.LeakyReLU(0.2, True),
            nn.Conv2d(144, 144, 3, stride=1, padding=1),  nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(144),
            nn.Conv2d(144, 256, 3, stride=2, padding=1),  nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),  nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),  nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),  nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),  nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),  nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(512),
        )
        self.decoder = ModSequential(
            nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1), nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1), nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 144, 3, stride=2, padding=1, output_padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(144, 144, 3, stride=1, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(144, 144, 3, stride=1, padding=1), nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(144),
            nn.ConvTranspose2d(144, 64, 5, stride=2, padding=2, output_padding=1),nn.LeakyReLU(0.2, True),
            nn.Conv2d(64,  16, 3, stride=1, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(16,  out_channels, 3, stride=1, padding=1)#,nn.LeakyReLU(True)
            ,nn.Sigmoid()
        )


    def forward(self, x):
        batch_size,_,height,width = x.size()
        x = x - 0.5
        x = self.encoder(x)    
        
        x = self.decoder(x)
        return x


class Encoder_Classifier(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder_Classifier, self).__init__()
        
        self.FC1 = nn.Linear(512*36*36, 1000)
        self.FC2 = nn.Linear(1000, 100)
        self.FC3 = nn.Linear(100, 10)
        
        self.encoder = ModSequential(
            nn.Conv2d(in_channels, 128, 5, stride=2, padding=2),  nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 144, 3, stride=1, padding=1),  nn.LeakyReLU(0.2, True),
            nn.Conv2d(144, 144, 3, stride=1, padding=1),  nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(144),
            nn.Conv2d(144, 144, 3, stride=2, padding=1),  nn.LeakyReLU(0.2, True),
            nn.Conv2d(144, 144, 3, stride=1, padding=1),  nn.LeakyReLU(0.2, True),
            nn.Conv2d(144, 144, 3, stride=1, padding=1),  nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(144),
            nn.Conv2d(144, 256, 3, stride=2, padding=1),  nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),  nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),  nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),  nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),  nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),  nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(512),
        )
            


    def forward(self, x):
        batch_size,_,height,width = x.size()
        x = x - 0.5
        x = self.encoder(x)
        
        x = x.view(x.size()[0],-1)
        print(x.size())
        x = self.FC1(x)
        x = self.FC2(x)
        x = self.FC3(x)
        return x


def train_step(inputs, labels, optimizer, criterion, unet):
    optimizer.zero_grad()
    outputs = unet(inputs)
    print(outputs.size())
    print(labels.size())
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss


def test_step(inputs, labels, optimizer, criterion, unet):
    optimizer.zero_grad()
    print("Test step")
    print("inputs ",inputs.shape)
    outputs = unet(inputs)
    loss = criterion(outputs, labels)
    print(loss)
    for i in range(4):
        img =  np.transpose(inputs[i,:3,:,:].detach().cpu().clone().numpy(),(1,2,0))
        opt = np.transpose(outputs[i,:3,:,:].detach().cpu().clone().numpy(),(1,2,0))
        lbl = np.transpose(labels[i,:3,:,:].detach().cpu().clone().numpy(),(1,2,0))
        images=[img,opt,lbl]
        names = ["image","output","label"]
        for k in range(len(images)):
            imagelib.imsave(names[k]+"_"+str(i)+".png", images[k])
     return outputs


def val_step(inputs, labels,optimizer, criterion, unet):
    optimizer.zero_grad()
    print("Validation step")
    print("inputs ",inputs.shape)
    outputs = unet(inputs)
    loss = criterion(outputs, labels)
    return loss


learning_rate  = 0.001
network_momentum = 0.99

unet = Encoder_Decoder(3,3)
unet = unet.to(device)

if device == 'cuda':
    unet = torch.nn.DataParallel(unet)
    cudnn.benchmark = True
criterion = nn.L1Loss()
optimizer = optim.SGD(unet.parameters(), lr = learning_rate, momentum = network_momentum)

training_dataset = STL10DataSet(train_images[:20000])
training_dataset_size = training_dataset.__len__() 
indices = list(range(training_dataset_size))

validation_dataset = STL10DataSet(train_images[20000:20016])

train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True,num_workers=12)
print(len(train_loader))
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=4, shuffle=True,num_workers=12)
print(len(validation_loader))

t = trange(epochs, leave=True)
for iter in t:
    total_loss = 0
    prev = 0
    unet.train()
    print("\n\n\n******* epoch number = " + str(iter+1)+" *********\n\n\n")
    ep_start = time.time()
    for batch_index, (input_batch, labels) in enumerate(train_loader):
        if use_gpu:
                batch_x = input_batch.to(device)
                batch_y = labels.to(device)
                
        else:
                batch_x = input_batch
                batch_y = labels
        with torch.autograd.detect_anomaly():        
                batch_loss = train_step(batch_x , batch_y, optimizer, criterion, unet)
                total_loss += batch_loss
        end = time.time()
        prev = time.time()
        if batch_index%50==0:
            print("batch_training_loss = " ,(batch_loss))
    print("\n\n\n******** total_epoch_training_loss = " + str(total_loss/len(train_loader))+" ********\n\n\n")
    
    dest_model_path = 'model_'+str(iter+1)+'.pth'
    torch.save(unet.state_dict(), dest_model_path)
    
    print("\n\n\n******** VALIDATION STEP STARTED ********\n\n\n")
    
    total_val_loss = 0
    
    unet.eval()
    
    with torch.no_grad():
        for batch_index, (input_batch, labels,actual_image_region) in enumerate(validation_loader):
            print("validation_batch_index = " + str(batch_index))
            if use_gpu:
                    batch_x = input_batch.to(device)
                    batch_y = labels.to(device)
                    batch_roi= actual_image_region.to(device)
            else:
                    batch_x = input_batch
                    batch_y = labels
                    batch_roi= actual_image_region
            with torch.autograd.detect_anomaly():        
                    batch_loss = val_step(batch_x , batch_y, batch_roi, optimizer, criterion, unet)
                    total_val_loss += batch_loss
            end = time.time()
            print("batch_time = " + str(end-prev))
            prev = time.time()
            print("batch_validation_loss = " + str(batch_loss))
              
    
    print("\n\n\n******** total_epoch_validation_loss = " + str(total_val_loss/len(validation_loader))+" ********\n\n\n")

torch.save(unet.state_dict(), 'model_final.pth')


classifier = Encoder_Classifier(3,10)

state_dict = torch.load("model_final.pth")
state_dict_modified = {}
for key in state_dict.keys():
    modified_key = key.split('.',1)[1]
    state_dict_modified[modified_key] = state_dict[key]
classifier.load_state_dict(state_dict_modified,strict=False)
classifier = classifier.cuda()

data_transforms = transforms.Compose([
    transforms.Resize((568,568)),
    transforms.ToTensor()
])


train_loader_classifier = torch.utils.data.DataLoader(
    datasets.ImageFolder('data' + '/train_images',
                         transform=data_transforms),
    batch_size=1, shuffle=True, num_workers=1)

val_loader_classifier = torch.utils.data.DataLoader(
    datasets.ImageFolder('data' + '/val_images',
                         transform=data_transforms),
    batch_size=1, shuffle=False, num_workers=1)

dataloaders_dict = {'train' : train_loader_classifier, 'val' : val_loader_classifier}

criterion_classification = nn.CrossEntropyLoss()
optimizer_classification =optim.SGD(classifier.parameters(), lr = learning_rate, momentum = network_momentum)

epoch_classification = 10
for iter in range(epoch_classification):
    total_loss = 0
    prev = 0
    classifier.train()
    print("\n\n\n******* epoch number = " + str(iter+1)+" *********\n\n\n")
    ep_start = time.time()
    for batch_index, (input_batch, labels) in enumerate(train_loader_classifier):
        if use_gpu:
                batch_x = input_batch.to(device)
                batch_y = labels.to(device)
                
        else:
                batch_x = input_batch
                batch_y = labels
        with torch.autograd.detect_anomaly():        
                batch_loss = train_step(batch_x , batch_y, optimizer_classification, criterion_classification, classifier)
                total_loss += batch_loss
        end = time.time()
        prev = time.time()
        if batch_index%50==0:
            print("batch_training_loss = " ,(batch_loss))
        
    print("\n\n\n******** total_epoch_training_loss = " + str(total_loss/len(train_loader))+" ********\n\n\n")
    
    dest_model_path = 'flow_sp_dropout'+str(iter+1)+'.pth'
    torch.save(unet.state_dict(), dest_model_path)
