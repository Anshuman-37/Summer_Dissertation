
## Author - M.Sc. Machine Learning in Sciences Anshuman Singh - ppxas6@nottingham.ac.uk
## Date - 21/06/2022 
## Title - Predicting Cereberal Blood Flow - Summer Disertation 2022 MLIS 


### Import Files
import torch
print(torch.__version__)
print(torch.cuda.is_available())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, LazyConv3d , MaxPool3d, Module, Softmax, BatchNorm3d, Dropout, Conv3d, MSELoss
from torch.optim import Adam, SGD
from torchsummary import summary as tfsum 
from tqdm import tqdm
torch.cuda.empty_cache()



### MODEL - 1 BaseLine
hidden = lambda c_in, c_out: Sequential(
    Conv3d(c_in, c_out, (31,31,25)),
    BatchNorm3d(c_out),
    ReLU(),
    MaxPool3d(1)
    )
Hidden = lambda c_in, c_out: Sequential(
    Conv3d(c_in, c_out, (29,29,24)),
    BatchNorm3d(c_out),
    ReLU(),
    MaxPool3d(1)
    )

class BaseModel(Module):
    '''Class for the Model to be fitted on MRI data'''
    def __init__(self, c):
        '''Intiallizing the layers of the Model'''
        super(BaseModel, self).__init__();
        self.hidden1 = hidden(1,c);
        self.hidden2 = hidden(c,2*c);
        self.hidden3 = hidden(2*c,4*c);
        self.hidden4 = hidden(4*c, 8*c);
        self.hidden5 = hidden(8*c, 4*c);
        self.hidden6 = hidden(4*c,2*c);
        self.hidden7 = Hidden(2*c,1);
    def forward(self, x):
        '''Implements the forward pass of the Network'''
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        x = self.hidden6(x)
        x = self.hidden7(x)
        return x 
    def summary(self):
        return tfsumm(self,(1, 91, 109, 91))

channels = 1; #3 Defining the number od channels we have for neural network
# Intializing model

def get_model(Name,channels):
    '''
    Params - Name of the model
    Result - Returns the model
    '''
    if Name == 'Base':
        return BaseModel(channels).to(device)
        
# def get_summary(model):
#     '''
#     Params - Pytorch Model
#     Result - Returns the summary of the model
#     '''        
#     return summary(model,(1, 288, 288, 180))

def train(model,epochs,batch_size,x,y):
        '''
        Params - Gets the model being trained , epochs, batch size , train x  and train y 
        Retunrs the train loss and the To train the neural network and return the losses
        '''
        ctr = 0; train_loss = []; 
        lossFn = MSELoss(); opt = Adam(model.parameters(), lr=1e-5);
        X = x.to(device); Y = y.to(device);
        for e in tqdm(range(0, epochs)):
            model.train(); batch_loss = [];
            permutation = torch.randperm(X.size()[0])
            for i in range(0,X.size()[0], batch_size):
                opt.zero_grad();
                indices = permutation[i:i+batch_size];
                batch_x, batch_y = X[indices], Y[indices];
                pred = model(batch_x) ; loss = lossFn(pred, batch_y);
                loss.backward(); opt.step();
                batch_loss.append(loss); 
            train_loss.append(batch_loss);
            print(batch_loss); ctr = ctr+1; 
            if ctr < epochs-1:
                del loss , pred
        return train_loss, pred; 