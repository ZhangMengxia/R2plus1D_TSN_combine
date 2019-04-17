import os
import time

import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

def test_model(model, val_dataloader, path="model_data.pth"):
    """Test a model, 
        Args:
            model (nn.Module): Model for the task
            val_dataloader (): data loader for validation
            path (str, optional): The directory to load a model checkpoint from.
    """


    model = model.to(device)
    criterion = nn.CrossEntropyLoss() # standard crossentropy loss for classification


    dataset_size = len(val_dataloader.dataset) 

    # saves the time the process was started, to compute total time at the end
    start = time.time()
    epoch_resume = 0

    # check if there was a previously saved checkpoint
    if os.path.exists(path):
        # loads the checkpoint
        checkpoint = torch.load(path)
        print("Reloading from previously saved checkpoint")

        # restores the model and optimizer state_dicts
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("Checkpoint does not exists {}".format(path))
        return


    # reset the running loss and corrects
    running_loss = 0.0
    running_corrects = 0

    model.eval()


    for inputs, labels in val_dataloader:
        # move inputs and labels to the device the training is taking place on
        inputs = inputs.to(device)
        labels = labels.to(device)

        # keep intermediate states iff backpropagation will be performed. If false, 
        # then all intermediate states will be thrown away during evaluation, to use
        # the least amount of memory possible.
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            # we're interested in the indices on the max values, not the values themselves
            _, preds = torch.max(outputs, 1)  
            loss = criterion(outputs, labels)


        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects.double() / dataset_size
    time_val = time.time() - start
    time_per_video = time_val / dataset_size
    print("Time: {}s Each video takes {}s Loss: {} Acc: {}".format(time_val, time_per_video, epoch_loss, epoch_acc))

def train_model(model, train_dataloader, val_dataloader, num_epochs=45, save=True, path="model_data.pth"):
    """Initalizes and the model for a fixed number of epochs, using dataloaders from the specified directory, 
    selected optimizer, scheduler, criterion, defualt otherwise. Features saving and restoration capabilities as well. 
    Adapted from the PyTorch tutorial found here: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

        Args:
            model (nn.Module): Model for the task
            train_dataloader (): data loader for training
            val_dataloader (): data loader for validation
            num_epochs (int, optional): Number of epochs to train for. Defaults to 45. 
            save (bool, optional): If true, the model will be saved to path. Defaults to True. 
            path (str, optional): The directory to load a model checkpoint from, and if save == True, save to. Defaults to "model_data.pth.tar".
    """


    model = model.to(device)
    criterion = nn.CrossEntropyLoss() # standard crossentropy loss for classification
    optimizer = optim.SGD(model.parameters(), lr=0.01)  # hyperparameters as given in paper sec 4.1
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}

    # saves the time the process was started, to compute total time at the end
    start = time.time()
    epoch_resume = 0

    # check if there was a previously saved checkpoint
    if os.path.exists(path):
        # loads the checkpoint
        checkpoint = torch.load(path)
        print("Reloading from previously saved checkpoint")

        # restores the model and optimizer state_dicts
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])
        
        # obtains the epoch the training is to resume from
        epoch_resume = checkpoint["epoch"]

    for epoch in tqdm(range(epoch_resume, num_epochs), unit="epochs", initial=epoch_resume, total=num_epochs):
        # each epoch has a training and validation step, in that order
        for phase in ['train', 'val']:

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0

            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':
                # scheduler.step() is to be called once every epoch during training
                scheduler.step()
                model.train()
            else:
                model.eval()


            for inputs, labels in dataloaders[phase]:
                # move inputs and labels to the device the training is taking place on
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                # keep intermediate states iff backpropagation will be performed. If false, 
                # then all intermediate states will be thrown away during evaluation, to use
                # the least amount of memory possible.
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    # we're interested in the indices on the max values, not the values themselves
                    _, preds = torch.max(outputs, 1)  
                    loss = criterion(outputs, labels)

                    # Backpropagate and optimize iff in training mode, else there's no intermediate
                    # values to backpropagate with and will throw an error.
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()   

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print("{} Loss: {} Acc: {}".format(phase, epoch_loss, epoch_acc))

        # save the model if save=True
        if save:
            torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': epoch_acc,
            'opt_dict': optimizer.state_dict(),
            }, path)

    # print the total time needed, HH:MM:SS format
    time_elapsed = time.time() - start    
    print("Training complete in {}h {}m {}s".format(time_elapsed//3600,
                    (time_elapsed%3600)//60, (time_elapsed %60)))
