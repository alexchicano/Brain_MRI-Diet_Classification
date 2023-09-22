from metadata import *
import torch
import torchvision.models as models
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from scipy import ndimage
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Variable
import os


# Create a custom dataset class
class CustomImageDataset(Dataset):
    def __init__(self, mtd, IDs, transform=None, interval = [], binarize = False):
        id_dict = mtd.id_dict
        self.transform = transform
        self.image_paths = [id_dict[id]['preprocessed'] for id in IDs]
        self.scores = [id_dict[id]['score'] for id in IDs]
        self.indices = [mtd.df.loc[mtd.df['ID'] == id].index[0] for id in IDs]
        
        # if bins is not empty, then we want to binarize the scores
        if len(interval) != 0 and binarize == True:
            # Binarize the scores:
            for i in range(len(self.scores)):
                # iterate over "scores" and assign "i" to the self.scores that are in the interval[i][0] and interval[i][1] range:
                for j in range(len(interval)):
                    if self.scores[i] >= interval[j][0] and self.scores[i] <= interval[j][1]:
                        self.scores[i] = j
                        break
        # "else" case is non-empty interval and binarize = False, which means we want to keep the scores as they are
        
    def __len__(self):
        return len(self.scores)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = nib.load(img_path).get_fdata()
        
        label = self.scores[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def get_balanced_sampler(mtd):
    scores = mtd.scores
    IDs = mtd.IDs

    # Set the ratio of the train, validation and test set: (20% validation, 20% test)
    r_train, r_val = [.6 , .2 ]

    # Create a data structure to organize the IDs by score:
    values = np.unique(scores)
    score_IDs = {value: [] for value in values}

    # loop over IDs and add them to the corresponding score:
    for id in IDs:
        score_id = mtd.df[mtd.df['ID']== id]['score'].values[0]
        score_IDs[score_id].append(id)

    score_IDs = {value: np.array(IDs) for value, IDs in score_IDs.items()}

    # Loop over the values and create a balanced train, validation and test set:
    train_IDs, val_IDs, test_IDs = [], [], []
    for value in values:
        IDs = score_IDs[value]
        len_IDs = len(IDs)

        train_IDs.extend(IDs[:round(r_train*len_IDs)])
        val_IDs.extend(IDs[round(r_train*len_IDs):round((r_train+r_val)*len_IDs)])
        test_IDs.extend(IDs[round((r_train+r_val)*len_IDs):])

    return train_IDs, val_IDs, test_IDs


def get_data(data_transforms, batch_size, num_workers, interval = [], binarize = False):
    
    mtd = DatasetMetadata( 'ImaGenoma', 'T1_b', interval = interval)
    # Get only a dataset for a specific interval
    id_dict = mtd.id_dict
    IDs = mtd.IDs
    
    # Read the id_dict[id][score] list and split it into train, val, and test sets:
    train_IDs, val_IDs, test_IDs = get_balanced_sampler(mtd)

    train_dataset =  CustomImageDataset(mtd, train_IDs, transform=data_transforms, interval=interval, binarize=binarize)
    val_dataset =  CustomImageDataset(mtd, val_IDs, transform=data_transforms, interval=interval, binarize=binarize)
    test_dataset =  CustomImageDataset(mtd, test_IDs, transform=data_transforms, interval=interval, binarize=binarize)

    splits = ['train', 'val', 'test']

    datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) for x in splits}

    return dataloaders, mtd, IDs, id_dict


def resize_volume(img, out_shape):
    
    desired_depth = out_shape[0]
    desired_width = out_shape[1]
    desired_height = out_shape[2]

    current_depth = img.shape[0]
    current_width = img.shape[1]
    current_height = img.shape[2]
 
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    
    img = ndimage.zoom(img, (depth_factor, width_factor, height_factor), order=1)
    return img

def imshow(img):
    # functions to show an image
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg[:,:,100], cmap='gray')
    plt.show()


def save_test_predictions(net, dataloaders, mtd, csv_path):
    #obtain the batch size from the dataloader
    batch_size = dataloaders['test'].batch_size

    # make a csv file with the predictions for the test set and their labels
    test_df = pd.DataFrame(columns=['ID', 'Predicted', 'Label'])

    # get the indices of the test set in batch_size chunks
    indices = dataloaders['test'].dataset.indices
    indices = [indices[i:i+batch_size] for i in range(0, len(indices), batch_size)]
    net.cpu()
    with torch.no_grad():
        # obtain the predictions for the test set
        for i, data in tqdm(enumerate(dataloaders['test'])):
            batch_indices = indices[i]
            inputs, labels = data
            inputs = inputs.unsqueeze(1).float()
            labels = labels.float()
            
            predicted = net(inputs)
            predicted = predicted.detach().numpy()
            labels = labels.detach().numpy()
            
            if len(predicted.shape) == 2:
                predicted = np.argmax(predicted, axis=1)
                for j in range(len(predicted)):
                    ID = mtd.df.loc[batch_indices[j]].ID
                    test_df.loc[len(test_df)] = [ID, predicted[j], labels[j]]
            else:
                for j in range(len(predicted)):
                    ID = mtd.df.loc[batch_indices[j]].ID
                    test_df.loc[len(test_df)] = [ID, round(predicted[j]), round(labels[j])]    
    
    # save the csv file
    test_df.to_csv(csv_path)
    
    return test_df


def get_accuracy(df):
    # This is going to include more metrics in the future, for now it only computes the accuracy
    correct = 0
    for i in range(len(df)):
        if abs(df['Predicted'][i] - df['Label'][i]) < 1:
            correct += 1
    accuracy = correct / len(df)
    return accuracy    


def compute_loss(model, device, data, criterion = nn.MSELoss()):
    
    inputs, targets = data
    
    inputs = inputs.unsqueeze(1).float()
    inputs = inputs.to(device)
    
    targets = targets.float()
    targets = targets.to(device)

    outputs = model(inputs)
    
    if outputs.shape[1] > 1: # Classification
        _, outputs = torch.max(outputs, 1)
        outputs = outputs.view(-1) # Convert to same size as labels
        print(outputs.item(), '(',int(targets.item()),')')
        loss = criterion(outputs.float(), targets)
        loss = Variable(loss, requires_grad = True)
    else:
        outputs = outputs.view(-1) # Convert to same size as labels
        loss = criterion(outputs, targets)
        
    return loss


def save_losses(model_name, loss_train, loss_val):
    # Create a "losses.csv" in the results folder, if the csv doesn't exist
    if not os.path.exists('../results/losses.csv'):
        df = pd.DataFrame(columns=['model_name', 'train_loss', 'val_loss'])
        df.to_csv('../results/losses.csv')

    # save the losses as a csv file
    df = pd.read_csv('../results/losses.csv', index_col=0)
    df.loc[len(df)] = [model_name, loss_train, loss_val]
    df.to_csv(f'../results/losses.csv')


def get_test_loss(model, dataloaders, criterion):
    # compute the loss on the test set
    with torch.no_grad():
        test_loss = 0
        for data in dataloaders['test']:
            loss = compute_loss(model, 'cpu', data, criterion)
            test_loss = test_loss + loss.item()
    return test_loss/len(dataloaders['test'])


def get_sensitivity_specificity(df):
    tp = df[(df['Predicted'] == 1) & (df['Label'] == 1)].shape[0]
    fn = df[(df['Predicted'] == 0) & (df['Label'] == 1)].shape[0]
    tn = df[(df['Predicted'] == 0) & (df['Label'] == 0)].shape[0]
    fp = df[(df['Predicted'] == 1) & (df['Label'] == 0)].shape[0]
    
    if tp + fn == 0:
        sensitivity = 0
    else:
        sensitivity = tp / (tp + fn)
        
    if tn + fp == 0:
        specificity = 0
    else:
        specificity = tn / (tn + fp)
    
    return sensitivity, specificity


def train_model(model, device, dataloaders, criterion, optimizer, num_epochs, patience, print_every, SAVE_PATH):
    train_losses = []
    val_losses = []
    
    # This sets the model in evaluation mode, which is disables dropout and batch normalization
    model.eval()
    
    print('Computing initial loss...')
    # start by calculating the loss on the validation set
    with torch.no_grad():
        best_val_loss = 0.0
        for data in tqdm(dataloaders['val']):
            loss = compute_loss(model, device, data, criterion)
            best_val_loss += loss.item()
        
    best_val_loss /= len(dataloaders['val'])
    print('Initial validation loss: {:.4f}'.format(best_val_loss))
    
    
    print('Training model...')
    counter = 0
    # Train your model
    for epoch in range(num_epochs):
        
        # Now model is in training mode, enabling dropout and batch normalization
        model.train()
        
        train_loss = 0.0
        for i, data in enumerate(dataloaders['train'], 0):
            loss = compute_loss(model, device, data, criterion)
            loss.backward()
            optimizer.step()
            
            # print statistics
            train_loss += loss.item()
            
            if i % print_every == 0:    # print every "print_every" mini-batches
                print(f'[{epoch + 1}, {i}] loss: {train_loss / (i+1):.3f}')
                
        # store the loss:
        train_loss = train_loss / len(dataloaders['train'])
        train_losses.append(train_loss)
        
        # model is put in evaluation mode again and the validation loss is computed
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for data in dataloaders['val']:
                loss = compute_loss(model, device, data, criterion)
                val_loss += loss.item()
            val_loss /= len(dataloaders['val'])
            # store the loss:
            val_losses.append(val_loss)
            
            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                torch.save(model.state_dict(), f'{SAVE_PATH}')
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping criteria met, val_loss: {best_val_loss}")
                    torch.save(model.state_dict(), f'{SAVE_PATH[:-4]}'+'_early_stopping.pth')
                    break
                print('Patience_left = ', patience-counter)
                
        print('Epoch:', epoch+1 , 'Train Loss:', train_loss, 'Val Loss:', val_loss, 'Best Loss:' , best_val_loss)
        # Check for early stopping
        if counter >= patience:
            break
    
        # Print a space between epochs
        print()
        
    print('Finished Training')
    
    return  train_losses, val_losses


class Net3c2d(nn.Module):
    # 3 convolutional layers, 2 dense layers
    def __init__(self):
        
        super().__init__()
        self.conv1 = self._conv_block(1, 64)    # 1 input channel, 64 output channels, 3x3 kernel
        self.conv2 = self._conv_block(64, 128)  # 64 input channels, 128 output channels, 3x3 kernel
        self.dropout_conv2 = nn.Dropout3d(p=0.3)    
        self.conv3 = self._conv_block(128, 128) # 128 input channels, 128 output channels, 3x3 kernel
        self.AP = nn.AvgPool3d((2, 2, 2))       # 2x2x2 kernel
        self.fc1 = self._dense_block(3456, 128) # 2**3*128 input channels, 128 output channels
        self.dropout_fc1 = nn.Dropout(p=0.3)
        self.fc2 = self._dense_block(128, 1)
        self.float()
        
    def _dense_block(self, in_c, out_c):
        dense_layer = nn.Sequential(
        nn.Linear(in_c, out_c),
        nn.ReLU()
        )
        return dense_layer
    
    def _conv_block(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
        nn.ReLU(),
        nn.MaxPool3d((3, 3, 3)),
        nn.BatchNorm3d(out_c)
        )
        return conv_layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout_conv2(x)
        x = self.conv3(x)
        x = self.AP(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.dropout_fc1(x)
        x = self.fc2(x)
        
        return x


class Net3c2d_bin(nn.Module):
    # 3 convolutional layers, 2 dense layers
    def __init__(self):
        
        super().__init__()
        self.conv1 = self._conv_block(1, 64)    # 1 input channel, 64 output channels, 3x3 kernel
        self.conv2 = self._conv_block(64, 128)  # 64 input channels, 128 output channels, 3x3 kernel
        self.dropout_conv2 = nn.Dropout3d(p=0.3)    
        self.conv3 = self._conv_block(128, 128) # 128 input channels, 128 output channels, 3x3 kernel
        self.AP = nn.AvgPool3d((2, 2, 2))         # 2x2x2 kernel
        self.fc1 = self._dense_block(3456, 128) # 2**3*128 input channels, 128 output channels
        self.dropout_fc1 = nn.Dropout(p=0.3)
        self.fc2 = self._dense_block(128, 2)
        self.float()
        
    def _dense_block(self, in_c, out_c):
        dense_layer = nn.Sequential(
        nn.Linear(in_c, out_c),
        nn.ReLU()
        )
        return dense_layer
    
    def _conv_block(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
        nn.ReLU(),
        nn.MaxPool3d((3, 3, 3)),
        nn.BatchNorm3d(out_c)
        )
        return conv_layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout_conv2(x)
        x = self.conv3(x)
        x = self.AP(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.dropout_fc1(x)
        x = self.fc2(x)
        
        return x


class ResNet18(nn.Module):
    """ This implementation loads a pretrained ResNet18 model and then creates a modified version of it."""
    
    def __init__(self, output_size, pretrained=False, freeze_weights=True):
        super().__init__()
        
        # Load the architecture and weights of the pretrained model:
        if pretrained == 'Kinetics':
            r3d_18 = models.video.r3d_18(weights='R3D_18_Weights.KINETICS400_V1')
            
            # Replace the first convolutional layer with a new one that has 1 input channel:
            weight = torch.zeros(1,1,3,7,7)
            for i in range(3):
                weight += r3d_18.stem[0].weight[:,i,:,:,:]**2
            weight = torch.sqrt(weight)
            r3d_18.stem[0] = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
            r3d_18.stem[0].weight = nn.Parameter(weight)
        else:
            r3d_18 = models.video.r3d_18()
            r3d_18.stem[0] = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
            if pretrained == 'MedicalNet':
                self.update_MedicalNet_weights(r3d_18)
        
        
        
        # Extract all layers from the pretrained model and modify the sizes of the input and output layers:
        self.stem = r3d_18.stem
        
        self.layer1 = r3d_18.layer1
        self.layer1 = r3d_18.layer1
        self.layer2 = r3d_18.layer2
        self.layer3 = r3d_18.layer3
        self.layer4 = r3d_18.layer4
        self.avgpool = r3d_18.avgpool
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)
        
        # If pretrained and desired, freeze the weights of the pretrained model to avoid changing the features extracted by it
        if freeze_weights and pretrained:
            parameters_to_freeze = [self.stem.parameters(), self.layer1.parameters(), self.layer2.parameters(), self.layer3.parameters()]
            for layer in parameters_to_freeze:
                for p in layer:
                    p.requires_grad = False
        
        # placeholder for the gradients
        self.gradient = None
        self.float()

    def update_MedicalNet_weights(self, r3d18):
        """Load the weights from the Medicalnet rn18_dict onto the r3d_18 model."""
        
        rn18_dict = torch.load('../models/pretrained/resnet_18.pth')['state_dict']
        
        # Copy the weights onto the convolutional layer of the stem
        r3d18.stem[0].weight = nn.Parameter(rn18_dict['module.conv1.weight'])

        # Copy the weights onto the batch normalization layer of the stem
        r3d18.stem[1].weight = nn.Parameter(rn18_dict['module.bn1.weight'])
        r3d18.stem[1].bias = nn.Parameter(rn18_dict['module.bn1.bias'])
        r3d18.stem[1].running_mean = nn.Parameter(rn18_dict['module.bn1.running_mean'])
        r3d18.stem[1].running_var = nn.Parameter(rn18_dict['module.bn1.running_var'])

        # Now copy the weights onto the other 4 layers
        # conv: r3d_18.layer{i=1,...,4}[j=0,1].conv{k=1,2}[0].weight = 
        #           nn.Parameter(rn18_dict[f'module.layer{i}.{j}.conv{k}.weight'])

        # bn:   r3d_18.layer{i=1,...,4}[j=0,1].conv{k=1,2}[1].{weight, bias, running_mean, running_var} = 
        #           nn.Parameter(rn18_dict[f'module.layer{i}.{j}.bn{k}.{weight, bias, running_mean, running_var}'])

        for i in range(1,4):
            for j in range(2):
                for k in range(1,3):
                    layer_i_j = getattr(r3d18, f'layer{i}')[j]
                    
                    # Copy the weights onto the convolution of the convolutional layer
                    conv_k = getattr(layer_i_j, f'conv{k}')[0]
                    conv_k.weight = nn.Parameter(rn18_dict[f'module.layer{i}.{j}.conv{k}.weight'])
                    
                    # Copy the weights onto the batch normalization of the convolutional layer
                    bn_k = getattr(layer_i_j, f'conv{k}')[1]
                    bn_k.weight = nn.Parameter(rn18_dict[f'module.layer{i}.{j}.bn{k}.weight'])
                    bn_k.bias = nn.Parameter(rn18_dict[f'module.layer{i}.{j}.bn{k}.bias'])
                    bn_k.running_mean = nn.Parameter(rn18_dict[f'module.layer{i}.{j}.bn{k}.running_mean'])
                    bn_k.running_var = nn.Parameter(rn18_dict[f'module.layer{i}.{j}.bn{k}.running_var'])
        return  

    # method for the gradient extraction
    def activations_hook(self, grad):
        self.gradient = grad
    
    def get_gradient(self):
        return self.gradient
    
    # method for the activation exctraction
    def get_activations(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
    def forward(self, x):
        
        # Pass through features layers
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
        
        
        # Pass through classification layers
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x


class ResNet18_1fc(nn.Module):
    """ This implementation loads a pretrained ResNet18 model and then creates a modified version of it."""
    
    def __init__(self, output_size, pretrained=False, freeze_weights=True):
        super().__init__()
        
        # Load the architecture
        if pretrained == 'Kinetics':
            r3d_18 = models.video.r3d_18(weights='R3D_18_Weights.KINETICS400_V1')
        else:
            r3d_18 = models.video.r3d_18()
        
        # Extract all layers from the pretrained model and modify the sizes of the input and output layers:
        self.stem = r3d_18.stem
        
        # Create a new Conv3d layer with the following original parameters and the new quadratic norm as weights
        self.stem[0] = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        
        self.layer1 = r3d_18.layer1
        self.layer1 = r3d_18.layer1
        self.layer2 = r3d_18.layer2
        self.layer3 = r3d_18.layer3
        self.layer4 = r3d_18.layer4
        self.avgpool = r3d_18.avgpool
        
        # Update all the weights of the model:
        if pretrained == 'MedicalNet':
            self.update_MedicalNet_weights()
        
        # We freeze the weights of the pretrained model to avoid changing the features extracted by it
        if freeze_weights and pretrained != False:
            self.stem.requires_grad_(False)
            self.layer1.requires_grad_(False)
            self.layer2.requires_grad_(False)
            self.layer3.requires_grad_(False)
            #self.layer4.requires_grad_(False)
        
        self.fc1 = nn.Linear(512, output_size)
        
        # placeholder for the gradients
        self.gradient = None
    
        self.float()

    def update_MedicalNet_weights(self):
        """Load the weights from the Medicalnet rn18_dict onto the r3d_18 model."""
        
        rn18_dict = torch.load('../models/pretrained/resnet_18.pth')['state_dict']
        
        # Copy the weights onto the convolutional layer of the stem
        self.stem[0].weight = nn.Parameter(rn18_dict['module.conv1.weight'])

        # Copy the weights onto the batch normalization layer of the stem
        self.stem[1].weight = nn.Parameter(rn18_dict['module.bn1.weight'])
        self.stem[1].bias = nn.Parameter(rn18_dict['module.bn1.bias'])
        self.stem[1].running_mean = nn.Parameter(rn18_dict['module.bn1.running_mean'])
        self.stem[1].running_var = nn.Parameter(rn18_dict['module.bn1.running_var'])

        # Now copy the weights onto the other 4 layers
        # conv: r3d_18.layer{i=1,...,4}[j=0,1].conv{k=1,2}[0].weight = 
        #           nn.Parameter(rn18_dict[f'module.layer{i}.{j}.conv{k}.weight'])

        # bn:   r3d_18.layer{i=1,...,4}[j=0,1].conv{k=1,2}[1].{weight, bias, running_mean, running_var} = 
        #           nn.Parameter(rn18_dict[f'module.layer{i}.{j}.bn{k}.{weight, bias, running_mean, running_var}'])

        for i in range(1,4):
            for j in range(2):
                for k in range(1,3):
                    layer_i_j = getattr(self, f'layer{i}')[j]
                    
                    # Copy the weights onto the convolution of the convolutional layer
                    conv_k = getattr(layer_i_j, f'conv{k}')[0]
                    conv_k.weight = nn.Parameter(rn18_dict[f'module.layer{i}.{j}.conv{k}.weight'])
                    
                    # Copy the weights onto the batch normalization of the convolutional layer
                    bn_k = getattr(layer_i_j, f'conv{k}')[1]
                    bn_k.weight = nn.Parameter(rn18_dict[f'module.layer{i}.{j}.bn{k}.weight'])
                    bn_k.bias = nn.Parameter(rn18_dict[f'module.layer{i}.{j}.bn{k}.bias'])
                    bn_k.running_mean = nn.Parameter(rn18_dict[f'module.layer{i}.{j}.bn{k}.running_mean'])
                    bn_k.running_var = nn.Parameter(rn18_dict[f'module.layer{i}.{j}.bn{k}.running_var'])
        return 

    # method for the gradient extraction
    def activations_hook(self, grad):
        self.gradient = grad
    
    def get_gradient(self):
        return self.gradient
    
    # method for the activation exctraction
    def get_activations(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
    def forward(self, x):
        
        # Pass through features layers
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
        
        # Pass through classification layers
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        
        
        return x

class ResNet18_2fc(nn.Module):
    """ This implementation loads a pretrained ResNet18 model and then creates a modified version of it."""
    
    def __init__(self, output_size, pretrained=False, freeze_weights=True):
        super().__init__()
        
        # Load the architecture
        if pretrained == 'Kinetics':
            r3d_18 = models.video.r3d_18(weights='R3D_18_Weights.KINETICS400_V1')
        else:
            r3d_18 = models.video.r3d_18()
        
        # Extract all layers from the pretrained model and modify the sizes of the input and output layers:
        self.stem = r3d_18.stem
        
        # Create a new Conv3d layer with the following original parameters and the new quadratic norm as weights
        self.stem[0] = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        
        self.layer1 = r3d_18.layer1
        self.layer1 = r3d_18.layer1
        self.layer2 = r3d_18.layer2
        self.layer3 = r3d_18.layer3
        self.layer4 = r3d_18.layer4
        self.avgpool = r3d_18.avgpool
        
        # Update all the weights of the model:
        if pretrained == 'MedicalNet':
            self.update_MedicalNet_weights()
        
        # We freeze the weights of the pretrained model to avoid changing the features extracted by it
        if freeze_weights and pretrained != False:
            self.stem.requires_grad_(False)
            self.layer1.requires_grad_(False)
            self.layer2.requires_grad_(False)
            self.layer3.requires_grad_(False)
            #self.layer4.requires_grad_(False)
        
        self.fc1 = nn.Linear(512, 256)
        self.fc1 = nn.Linear(256, 2)
        
        # placeholder for the gradients
        self.gradient = None
    
        self.float()

    def update_MedicalNet_weights(self):
        """Load the weights from the Medicalnet rn18_dict onto the r3d_18 model."""
        
        rn18_dict = torch.load('../models/pretrained/resnet_18.pth')['state_dict']
        
        # Copy the weights onto the convolutional layer of the stem
        self.stem[0].weight = nn.Parameter(rn18_dict['module.conv1.weight'])

        # Copy the weights onto the batch normalization layer of the stem
        self.stem[1].weight = nn.Parameter(rn18_dict['module.bn1.weight'])
        self.stem[1].bias = nn.Parameter(rn18_dict['module.bn1.bias'])
        self.stem[1].running_mean = nn.Parameter(rn18_dict['module.bn1.running_mean'])
        self.stem[1].running_var = nn.Parameter(rn18_dict['module.bn1.running_var'])

        # Now copy the weights onto the other 4 layers
        # conv: r3d_18.layer{i=1,...,4}[j=0,1].conv{k=1,2}[0].weight = 
        #           nn.Parameter(rn18_dict[f'module.layer{i}.{j}.conv{k}.weight'])

        # bn:   r3d_18.layer{i=1,...,4}[j=0,1].conv{k=1,2}[1].{weight, bias, running_mean, running_var} = 
        #           nn.Parameter(rn18_dict[f'module.layer{i}.{j}.bn{k}.{weight, bias, running_mean, running_var}'])

        for i in range(1,4):
            for j in range(2):
                for k in range(1,3):
                    layer_i_j = getattr(self, f'layer{i}')[j]
                    
                    # Copy the weights onto the convolution of the convolutional layer
                    conv_k = getattr(layer_i_j, f'conv{k}')[0]
                    conv_k.weight = nn.Parameter(rn18_dict[f'module.layer{i}.{j}.conv{k}.weight'])
                    
                    # Copy the weights onto the batch normalization of the convolutional layer
                    bn_k = getattr(layer_i_j, f'conv{k}')[1]
                    bn_k.weight = nn.Parameter(rn18_dict[f'module.layer{i}.{j}.bn{k}.weight'])
                    bn_k.bias = nn.Parameter(rn18_dict[f'module.layer{i}.{j}.bn{k}.bias'])
                    bn_k.running_mean = nn.Parameter(rn18_dict[f'module.layer{i}.{j}.bn{k}.running_mean'])
                    bn_k.running_var = nn.Parameter(rn18_dict[f'module.layer{i}.{j}.bn{k}.running_var'])
        return 

    # method for the gradient extraction
    def activations_hook(self, grad):
        self.gradient = grad
    
    def get_gradient(self):
        return self.gradient
    
    # method for the activation exctraction
    def get_activations(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
    def forward(self, x):
        
        # Pass through features layers
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
        
        # Pass through classification layers
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        
        
        return x
