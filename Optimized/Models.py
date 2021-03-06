
## Author - M.Sc. Machine Learning in Sciences Anshuman Singh - ppxas6@nottingham.ac.uk
## Date - 21/06/2022 
## Title - Predicting Cereberal Blood Flow - Summer Disertation 2022 MLIS 


### Import Files
from tkinter import BASELINE
import torch
print(torch.__version__)
print(torch.cuda.is_available())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, LazyConv3d , MaxPool3d, Module, Softmax, BatchNorm3d, Dropout, Conv3d, MSELoss
from torch.optim import Adam, SGD
from torchsummary import summary 
from tqdm import tqdm
torch.cuda.empty_cache()

## Torch Essentails
torch.cuda.empty_cache()
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

### MODEL - 1 3d Convnet 
hidden = lambda c_in, c_out: Sequential(
    Conv3d(c_in, c_out, (3,3,2), padding=(1,1,1)),
    BatchNorm3d(c_out),
    ReLU(),
    MaxPool3d(1)
    )
hidden_2 = lambda c_in, c_out: Sequential(
    Conv3d(c_in, c_out, (3,3,5),padding=(1,1,1)),
    BatchNorm3d(c_out),
    ReLU(),
    MaxPool3d(1)
    )
Hidden = lambda c_in, c_out: Sequential(
    Conv3d(c_in, c_out, (1,1,4)),
    BatchNorm3d(c_out),
    ReLU(),
    MaxPool3d(1)
    )

class convnet_3D(Module):
    '''Class for the Model to be fitted on MRI data'''
    def __init__(self, c):
        '''Intiallizing the layers of the Model'''
        super(convnet_3D, self).__init__();
        self.hidden1 = hidden(1,4*c);
        self.hidden2 = hidden(4*c,8*c);
        self.hidden3 = hidden(8*c,16*c);
        self.hidden4 = hidden(16*c,8*c);
        self.hidden5 = hidden(8*c, 16*c);
        self.hidden6 = hidden_2(16*c,8*c);
        self.hidden7 = Hidden(8*c,1);
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

def train_model_3D(epochs,batch_size,x,y,channels=1):
    '''
    Params - Epochs, Batch Size, Data, Channels default = 1 
    Result - Returns the predictions and train loss 
    '''
    model = convnet_3D(channels).to(device)
    summary(model,(1, 91,109,91))
    ctr = 0; train_loss = []; f_loss = []; predic = [];

    ## Model criterion
    lossFn = MSELoss(); opt = Adam(model.parameters(), lr=1e-5);
    X = x.to(device); Y = y.to(device); model.train();

    ## Loop around the model
    for e in tqdm(range(0, epochs)):
        # Batch Loss and Train Loss 
        batch_loss = []; train_loss = [];
        # Permutation for the data
        permutation = torch.randperm(X.size()[0])
        ctr = ctr + 1;
        ## Loop over batches 
        for i in range(0,X.size()[0], batch_size):
            ## Intialize the optimizer
            opt.zero_grad();
            ## Setting up the indexes for batch
            indices = permutation[i:i+batch_size];
            batch_x, batch_y = X[indices], Y[indices];
            ## Making prediction and Observing the loss
            pred = model(batch_x) ; loss = lossFn(pred, batch_y);
            ##  Update the loss using gradient descent
            loss.backward(); opt.step();
            ## Saving the loss on cpu rather than gpu to optimize memory
            batch_loss.append(loss.cpu().detach().numpy()); 
            if ctr == epochs - 1 : predic.append(pred)
            ## Deleting the batch variables to optimize memoery
            del batch_x , batch_y; 
            ## Deleting the loss and the predictions made 
            del loss , pred;
        train_loss.append(batch_loss);
        ## Printing the verbose
        if(e%10 == 0): 
            print(batch_loss[-1]); print((torch.cuda.memory_allocated())/(1024*1024))
    ## Freeing up GPU memory
    del X,Y;
    return train_loss,predic
