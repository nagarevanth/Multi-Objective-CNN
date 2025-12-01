import torch
import torch.nn as nn

class CNN_AE(nn.Module):
    def __init__(self, in_channels=1, channel_list=[16, 32, 64], kernel_size=3, stride=1, padding=1):
        super(CNN_AE, self).__init__()
        
        encoder_layers = []
        input_channels = in_channels
        for output_channels in channel_list:
            encoder_layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding))
            encoder_layers.append(nn.ReLU())
            input_channels = output_channels
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        decoder_layers = []
        for output_channels in reversed(channel_list[:-1]):
            decoder_layers.append(nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding))
            decoder_layers.append(nn.ReLU())
            input_channels = output_channels
        
        decoder_layers.append(nn.ConvTranspose2d(input_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding))
        decoder_layers.append(nn.Sigmoid())  # Activation for output reconstruction
        
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, X):
        return self.encoder(X)

    def decode(self, X):
        return self.decoder(X)

    def forward(self, X):
        latent_rep = self.encode(X)
        reconst = self.decode(latent_rep)
        return reconst

# model = CNN_AE()
# dummy_input = torch.randn(64, 1, 28, 28)
# output = model(dummy_input)
# print("Output shape:", output.shape) 
