
## Author - M.Sc. Machine Learning in Sciences Anshuman Singh - ppxas6@nottingham.ac.uk
## Date - 27/06/2022 
## Title - Predicting Cereberal Blood Flow - Summer Disertation 2022 MLIS 

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation

def predictions_2D(y_target,y_predicted,sample_number):
    '''
    Params - Y_target(To be predicted) , Y_predicted(Model Predictions) , Sample_number - index of prediction
    Result - Plots image of the model predicted and actual predictoins 
    '''
    y_p = y_predicted[0][sample_number]
    y = y_target.cpu().detach().numpy()
    y_p = y_p.cpu().detach().numpy()
    print(y_p.shape); print(type(y_p))
    print(y.shape); print(type(y))

    plt.imshow(y_p[0,:,:],cmap = 'gray'); plt.show();
    plt.imshow(y[sample_number][0,:,:],cmap = 'gray'); plt.show();

def predictions_3D(y_target,y_predicted,sample_number):
    '''
    Params - Y_target(To be predicted) , Y_predicted(Model Predictions) , Sample_number - index of prediction
    Result - Saves a video of the 3D convolution 
    '''
    y_p = y_predicted[0][sample_number]; 
    y = y_target.cpu().detach().numpy(); y_p = y_p.cpu().detach().numpy()
    frames = [] # for storing the generated images
    fig = plt.figure(); 
    for i in range(0,y.shape[4]):
        ## Appendin the plots
        frames.append([plt.imshow(y_p[0,:,:,i], cmap=cm.Greys_r,animated=True)]);
    ani = animation.ArtistAnimation(fig, frames, interval=120, blit=True, repeat_delay=1000)
    ani.save('Result.mp4');
    