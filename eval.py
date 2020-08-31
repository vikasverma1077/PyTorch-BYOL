import torch
import sys
import yaml
from torchvision import transforms, datasets
import torchvision
import numpy as np
import os
#from sklearn import preprocessing
from torch.utils.data.dataloader import DataLoader

sys.path.append('../')
from models.resnet_base_network import ResNet18


batch_size = 512
data_transforms = torchvision.transforms.Compose([transforms.ToTensor()])


config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)


train_dataset = datasets.STL10('/home/vermavik/PyTorch-BYOL/Downloads/NoAug', split='train', download=False,
                               transform=data_transforms)

test_dataset = datasets.STL10('/home/vermavik/PyTorch-BYOL/Downloads/NoAug', split='test', download=False,
                               transform=data_transforms)


print("Input shape:", train_dataset[0][0].shape)


train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          num_workers=0, drop_last=False, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=batch_size,
                          num_workers=0, drop_last=False, shuffle=True)

device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'
encoder = ResNet18(**config['network'])
output_feature_dim = encoder.projetion.net[0].in_features


#load pre-trained parameters
load_params = torch.load(os.path.join('/home/vermavik/github/PyTorch-BYOL/checkpoints/model.pth'),
                         map_location=torch.device(torch.device(device)))

if 'online_network_state_dict' in load_params:
    encoder.load_state_dict(load_params['online_network_state_dict'])
    print("Parameters successfully loaded.")

# remove the projection head
encoder = torch.nn.Sequential(*list(encoder.children())[:-1])    
encoder = encoder.to(device)

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)

logreg = LogisticRegression(output_feature_dim, 10)
logreg = logreg.to(device)

def get_features_from_encoder(encoder, loader):
    
    x_train = []
    y_train = []

    # get the features from the pre-trained model
    for i, (x, y) in enumerate(loader):
        with torch.no_grad():
            feature_vector = encoder(x)
            x_train.extend(feature_vector)
            y_train.extend(y.numpy())

            
    x_train = torch.stack(x_train)
    y_train = torch.tensor(y_train)
    return x_train, y_train

encoder.eval()
x_train, y_train = get_features_from_encoder(encoder, train_loader)
x_test, y_test = get_features_from_encoder(encoder, test_loader)

if len(x_train.shape) > 2:
    x_train = torch.mean(x_train, dim=[2, 3])
    x_test = torch.mean(x_test, dim=[2, 3])
    
print("Training data shape:", x_train.shape, y_train.shape)
print("Testing data shape:", x_test.shape, y_test.shape)


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test):
    train = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)

    test = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test, batch_size=512, shuffle=False)
    return train_loader, test_loader

"""
scaler = preprocessing.StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train).astype(np.float32)
x_test = scaler.transform(x_test).astype(np.float32)
"""
x_mean = x_train.mean(axis=0)
x_std = x_train.std(axis=0)
x_train = x_train.sub_(x_mean).div_(x_std)
x_test = x_test.sub_(x_mean).div_(x_std)

train_loader, test_loader = create_data_loaders_from_arrays(x_train, y_train, x_test, y_test)


optimizer = torch.optim.Adam(logreg.parameters(), lr=3e-4)
criterion = torch.nn.CrossEntropyLoss()
eval_every_n_epochs = 10

for epoch in range(1000):
#     train_acc = []
    for x, y in train_loader:

        x = x.to(device)
        y = y.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()        
        
        logits = logreg(x)
        predictions = torch.argmax(logits, dim=1)
        
        loss = criterion(logits, y)
        
        loss.backward()
        optimizer.step()
    
    total = 0
    if epoch % eval_every_n_epochs == 0:
        correct = 0
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits = logreg(x)
            predictions = torch.argmax(logits, dim=1)
            
            total += y.size(0)
            correct += (predictions == y).sum().item()
            
        acc = 100 * correct / total
        print("Testing accuracy:", np.mean(acc))


