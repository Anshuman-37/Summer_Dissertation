import Models
import Optimizers
import Loss
import tensorflow as tf 
import nibabel as nib

## The class for data.. 
class model:
    def __init__(self):
        '''Intializing the function model object'''
        this.input_size = (64,72,64,2) 
        this.model = Models.get_model('unet',this.input_size)
        this.optimizer = Optimizers.get_optimizer('Adam')
        #this.loss = Loss.get_loss('masked_mse')
     
    
    def load_data(self,path):
        '''Function to load the data'''
        this.mask_obj = nib.load(path)
        mask = mask_obj.get_data()
        batch_mask = mask

        # Should be equall to something
        # this.y_true =  
    
    def run_model(self,model_name,input_size,learning_rate,optimizer_name):#,loss_name,):
        '''This function will run our model according to our requirements'''
        this.model =  Models.get_model(model_name,input_size)
        this.optimizer = Optimizers.get_optimizer(optimizer_name)
        #this.loss = Loss.get_loss(loss_name,this.y_true,this.y_pred)
        # BotherSome steps
        # this.model.complie(learning_rate)