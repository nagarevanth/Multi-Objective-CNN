import torch.nn as nn
import torch.nn.functional as ActFn

class customed_CNN(nn.Module):
    def __init__(self, list_in_channels, final_out_channel, img_size, padding=0, stride=1, filter_size=5, task='classification', num_classes=10, fn='Count', drop_rate=0.2):
        super(customed_CNN, self).__init__()
        self.task = task
        self.fn = fn
        self.n_conv = len(list_in_channels)
        
        # Use ModuleList for Conv and BatchNorm layers
        self.Conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        for i in range(self.n_conv-1):
            Conv_layer = nn.Conv2d(in_channels=list_in_channels[i], out_channels=list_in_channels[i+1], kernel_size=filter_size, padding=padding, stride=stride)
            bn_layer = nn.BatchNorm2d(list_in_channels[i+1])
            self.bn_layers.append(bn_layer)
            self.Conv_layers.append(Conv_layer)
            img_size = int((img_size + 2*padding - filter_size) / stride) + 1
            img_size = int(img_size / 2)

        final_conv_layer = nn.Conv2d(in_channels=list_in_channels[self.n_conv-1], out_channels=final_out_channel, kernel_size=filter_size)
        final_bn = nn.BatchNorm2d(final_out_channel)
        self.Conv_layers.append(final_conv_layer)
        self.bn_layers.append(final_bn)
        img_size = int((img_size + 2*padding - filter_size) / stride) + 1
        img_size = int(img_size / 2)
        self.img_size = img_size

        self.MaxPool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(img_size * img_size * final_out_channel, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, X):
        feature_maps = []  # List to store feature maps

        for i in range(self.n_conv):
            X = self.Conv_layers[i](X)
            X = ActFn.relu(self.bn_layers[i](X))
            feature_maps.append(X)  # Store feature map
            X = self.MaxPool(X)
        
        X = self.dropout(X)
        X = X.view(X.size(0), -1)
        X = ActFn.relu(self.fc1(X))

        # Task-specific output
        if self.fn == 'Count':
            y_hat = ActFn.softmax(self.fc2(X), dim=1) if self.task == 'classification' else self.fc2(X)
        elif self.fn == 'Pred':
            y_hat = self.fc2(X)
        
        return y_hat, feature_maps  # Return feature maps along with output
