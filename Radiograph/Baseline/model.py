import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from pytorch_pretrained_vit import ViT


class Encoder(nn.Module):
    def __init__(self,model_core,num_ftrs,hidden_dim=64,layer_dim=1,output_dim=1,dropout=0.1,device = "cpu"):
        super(Encoder, self).__init__()
        self.encoder_model = model_core

        if layer_dim == 1:
            hidden_dim = num_ftrs
        else:
            self.do = nn.Dropout(p=dropout).to(device)
        
        self.projs = []
        
        for i in range(layer_dim-1):
            if i==0:
                self.projs.append(nn.Linear(num_ftrs, hidden_dim).to(device))
            else:
                self.projs.append(nn.Linear(hidden_dim, hidden_dim).to(device))
                
        

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.layer_dim = layer_dim
        
        
        

       

    def forward(self, x ):
        out=self.encoder_model(x)
        if self.layer_dim ==1:
        
            out = self.fc(out)

            return out
        else:
            for i in range(self.layer_dim-1):
                out = self.do(self.projs[i](out))
            out = self.fc(out)

            return out

class Siam_encoder(torch.nn.Module):
    def __init__(self, model_core ,num_ftrs):
        super(Siam_encoder, self).__init__()

        self.encoder_model = model_core
        self.act = nn.ReLU()

        self.linear1 = nn.Linear(num_ftrs,1) 






    def forward(self, x):
        hid1 = self.encoder_model(x[:,0,:,:,:])
        hid2 = self.encoder_model(x[:,1,:,:,:])


        out1 = self.linear1(self.act(hid1))
        out2 = self.linear1(self.act(hid2))


        




        return out1,out2,hid1,hid2
class EncLSTMModel(nn.Module):
    def __init__(self,model_core,num_ftrs,hidden_dim=64,layer_dim=1,output_dim=1,n_times=2,dropout=0.1,mode = "prev",device = "cpu"):
        super(EncLSTMModel, self).__init__()
        self.encoder_model = model_core
        self.n_times=n_times
        self.lstm = nn.LSTM(num_ftrs, hidden_dim, layer_dim, batch_first=True,dropout=dropout)

        # Readout layer
        if mode == "prev":
            self.fc = nn.Linear(hidden_dim, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim*2, output_dim)

        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        
        self.mode = mode
        self.device = device
        
        

       

    def forward(self, x ):
        x_list = [0]*self.n_times
        #print(x[:,0,:,:,:].shape)
        for i in range(self.n_times):
            x_list[i]=self.encoder_model(x[:,i,:,:,:])
            #print(l.shape)
        X = torch.stack(x_list, dim = 1)
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)

        X, (hn, cn) = self.lstm(X, (h0.detach(), c0.detach()))
        
         
        # out.size() --> 100, 10
        if self.mode == "prev":
            out =  X[:,-1,:]
        elif self.mode == "concat":
            out =  X.reshape((X.shape[0],-1))
        else:
            print("ERROR ERROR ERROR ERROR")
        out = self.fc(out)
        
        return out

            

            
        
        
        
        
            
        












def get_model(tl_model,mode=None,image_size=1024,cropped_size=896,hidden_dim=512,layer_dim=1,dropout=0.1,output_dim=1,device="cpu",types="baseline"):
    # load the pretrained model, Resnet34 was used in the paper
    if tl_model == "Resnet34":
        model_ft = models.resnet34(pretrained=True)
        if image_size == 1024:
            model_ft.avgpool = nn.AvgPool2d(kernel_size=28, stride=1, padding=0)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential()
        
    


    model_ft = Encoder(model_ft, num_ftrs,hidden_dim,layer_dim,output_dim,dropout,device)
    

    return model_ft
        
    

