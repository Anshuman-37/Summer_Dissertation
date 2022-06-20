import numpy as np

def masked_mse(y_true, y_pred):
    '''This function will provide us with the masked mean square error'''
    print("Batch size is:")
    print(y_true.shape[0])
    sq = tf.square(y_true - y_pred)
    masked_sq = tf.multiply(sq, batch_mask)
    print("Masked square shape is:")
    print(masked_sq.shape)
    loss = tf.reduce_sum(masked_sq, axis=[1,2,3,4])
    print("Loss shape is:")
    print(loss.shape)
    return loss

