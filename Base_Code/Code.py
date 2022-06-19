import Models
import Optimizers
import Loss

## The class for data.. 
class model:
    def __init__():
        '''Intializing the function model object'''
        this.model = Models.get_model('unet')
        this.optimizer = Optimizers.get_optimizer('Adam')
        this.loss = Loss.get_loss('masked_mse')
    
    def models(name):
        '''Function to annotate the required model'''
        this.model = Models.get_model(name)
    
    def optimizer(name,lr_rate):
        '''Function to annotate the required optimizer'''
        this.optimizer = Optimizers.get_optimizer(name)
    
    def loss(name):
        '''Function to annotate the specific loss function'''
        this.loss = Loss.get_loss(name)

    def load_data(path):
        '''Function to load the data'''
        #this.data = 