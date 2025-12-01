import sys
sys.path.append('../')

from models.MLP.MLP import MLP

class AutoEncoders:
    def __init__(self, input_size, reduced_size, lr=0.01, num_epoch=1000, batch_size=16, act_fn='relu', opt_fn='MBGD'):
        
        self.encoder = MLP(input_size=input_size,output_size=reduced_size,act_fn='tanh',batch_size=batch_size,n_neuron=[128,64,64,32],task_type='regression',opt_fn=opt_fn)
        
        self.decoder = MLP(input_size=reduced_size, output_size=input_size, lr=lr, num_epoch=num_epoch, act_fn=act_fn, 
                           opt_fn=opt_fn, n_neuron=[128,64,64,32], batch_size=batch_size, task_type='regression')

    
    def fit(self, X_train,X_train_red,X_val,X_val_red):
        losses_encoder = self.encoder.fit(X_train,X_train_red,X_val,X_val_red)
        latent_representation = self.encoder.forward(X_train)
        print("done")
        losses_decoder = self.decoder.fit(latent_representation, X_train, X_val=X_val_red, y_val=X_val)
        
        return losses_encoder, losses_decoder

    
    def get_latent(self, X):

        latent_representation = self.encoder.forward(X)
        return latent_representation

    
    
    def reconstruct(self, X):

        latent_representation = self.encoder.forward(X)
        reconstructed = self.decoder.forward(latent_representation)
        return reconstructed
