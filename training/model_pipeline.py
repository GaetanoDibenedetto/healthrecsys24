import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch import nn
from tqdm import tqdm

from utils.utils import load_model_classification, path_model_checkpoint


def train(model, device, train_loader, optimizer, epoch, _loss_criteria=None):
    if _loss_criteria is not None:
        loss_criteria = _loss_criteria
    # Set the model to training mode
    model.train()
    train_loss = 0
    loss=torch.tensor([0.0])

    for batch_idx, (data, label) in enumerate((pbar := tqdm(train_loader))):
        data, label = data.to(device), label.to(device)
        # Reset the optimizer
        optimizer.zero_grad()

        # Push the data forward through the model layers
        output = model(data)
        # Get the loss
        loss = loss_criteria(output, label)

        # Keep a running total
        train_loss += loss.item()
        # Backpropagate
        loss.backward()
        optimizer.step()

        pbar.set_description(f'Epoch {epoch} Loss {loss.item():.3f}')
    # return average loss for the epoch
    avg_loss = train_loss / (batch_idx+1)
    print('Training set: Average loss: {:.6f}'.format(avg_loss))
    return avg_loss

def validation(model, device, test_loader):
    # Switch the model to evaluation mode (so we don't backpropagate or drop)
    model.eval()
    test_loss = 0
    correct = 0
    tested_sample = 0
    with torch.no_grad():
        batch_count = 0
        for batch_idx, (data, label) in enumerate((pbar := tqdm(test_loader))):
            tested_sample += len(data)
            batch_count += 1
            data, label = data.to(device), label.to(device)
            # Get the predicted classes for this batch
            output = model(data)

            # Calculate the loss for this batch
            # predicted_label = output.argmax(axis = 1)[:,None]
            # label = label.squeeze().float()
            # output = output.squeeze()
            loss = loss_criteria(output, label).item()
            test_loss += loss

            # Calculate the accuracy for this batch
            # _, predicted = torch.max(output.data, 1)
            predicted = torch.max(output, axis=1)[1]
            label = torch.max(label, axis=1)[1]
            # predicted = torch.round(output)
            correct += torch.sum(label==predicted).item()
            pbar.set_description(f'Validation Loss {loss:.3f} Accuracy {correct/tested_sample:.3f}')
    # Calculate the average loss and total accuracy for this epoch
    avg_loss = test_loss / batch_count
    print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    # return average loss for the epoch
    return avg_loss


def test(model, device, test_loader):
    truelabels = []
    predictions = []
    model.eval()
    with torch.no_grad():
        for data, label in test_loader:

            data, label = data.to(device), label.to(device)
            output = model(data)
            output = torch.max(output, axis=1)[1]
            label = torch.max(label, axis=1)[1]
            
            predictions.append(output)
            truelabels.append(label)


    truelabels = torch.cat(truelabels).to("cpu")
    predictions = torch.cat(predictions).to("cpu")
    return truelabels, predictions


def run_pipeline(
    model,
    trainloader,
    valloader,
    testloader,
    model_type="image",
    epochs=50,
    learning_rate=0.001,
    weight_loss=None,
    device="cuda",
):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    global loss_criteria

    if weight_loss is None:
        loss_criteria = nn.BCELoss()
    else:
        weight = torch.tensor(weight_loss).to(device)        
        loss_criteria = nn.BCELoss(weight=weight)
    epoch_nums = []
    training_loss = []
    validation_loss = []

    epochs = epochs
    print('Training on', device)
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        train_loss = train(model, device, trainloader, optimizer, epoch, loss_criteria)
        epoch_nums.append(epoch)
        training_loss.append(train_loss)

        if valloader is not None:
            val_loss = validation(model, device, valloader)
            validation_loss.append(val_loss)
            if val_loss <= min(validation_loss):
                best_model = model
                best_epoch = epoch
        else:
            if train_loss <= min(training_loss):
                best_model = model
                best_epoch = epoch

    print("Best model updated at epoch", best_epoch)
    save_model(best_model, model_type)
    return test(best_model, device, testloader)


def save_model(model, model_type="image"):
    assert model_type in ["image", "keypoint"]

    model_path = os.path.join(path_model_checkpoint, model_type)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(
        model,
        os.path.join(model_path, f"{datetime.now().strftime('%Y%m%d%H%M%S')}.pkl"),
    )


def test_load_model(testloader, model_path=None, device="cuda"): 
    model = load_model_classification(model_path)

    return test(model, device, testloader)
