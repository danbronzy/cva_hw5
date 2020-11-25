import torch
import scipy.io
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
import numpy as np

def trainModelSGD(dataLoader, model, epochs, learning_rate, valid_data = None):

    #GPU-ify if possible. Using Dataloader so thats done in the loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on device: {}".format(device.type))
    model.to(device)
    
    datasetSize = len(dataLoader.dataset)
    opt = optim.SGD(model.parameters(), lr=learning_rate)

    if valid_data is not None:
        #extract and gpu-ify
        valid_x = valid_data['x'].to(device)
        valid_y = valid_data['y'].to(device)

    losses = []
    accs = []
    losses_valid = []
    accs_valid = []
    for epoch in range(epochs):
        loss_epoch = 0
        acc_epoch = 0
        for xb, yb in dataLoader:
            #gpu-ify
            xb, yb = xb.to(device), yb.to(device)

            y_pred = model(xb)

            lossFunc = torch.nn.CrossEntropyLoss()
            loss = lossFunc(y_pred, yb.argmax(axis=1))
            #.item() needed so as not to save computational graph
            loss_epoch += loss.item()
            loss.backward()
            
            guess = y_pred.argmax(axis = 1)
            correct = yb.argmax(axis=1)
            acc_epoch += (guess == correct).count_nonzero()

            opt.step()
            opt.zero_grad()

        
        losses.append(loss_epoch/datasetSize)
        accs.append(acc_epoch/datasetSize)
        if epoch % 5 == 0:
            print("Epoch: {:02d} \t loss: {:.2f} \t acc: {:.2f}".format(epoch,loss_epoch/datasetSize,acc_epoch/datasetSize))
        
        #if there is validation data to process
        if valid_data is not None:
            with torch.no_grad():
                model.eval()
                valid_y_pred = model(valid_x)
                guess = valid_y_pred.argmax(axis = 1)
                correct = valid_y.argmax(axis=1)

                lossFunc = torch.nn.CrossEntropyLoss()
                loss_valid = lossFunc(valid_y_pred, correct).item()/len(valid_y)
                losses_valid.append(loss_valid)

                acc_valid = (guess == correct).count_nonzero()/valid_y.shape[0]
                accs_valid.append(acc_valid)
                if epoch % 5 == 0:
                    print("|--Validation \t loss: {:.2f} \t acc: {:.2f}".format(loss_valid,acc_valid))
                model.train()

    ax1 = plt.subplot(121)
    ax1.plot(range(epochs), accs)
    ax1.plot(range(epochs), accs_valid)
    ax1.set_xlabel("Epochs")
    ax1.set_xlim([0, epochs])
    ax1.set_ylabel("Accuracy")
    ax1.legend(["Test Data", "Validation Data"])

    ax2 = plt.subplot(122)
    ax2.plot(range(epochs), losses)
    ax2.plot(range(epochs), losses_valid)
    ax2.set_xlabel("Epochs")
    ax2.set_xlim([0, epochs])
    ax2.set_ylabel("Average Loss")
    ax2.legend(["Test Data", "Validation Data"])
    plt.show()
    return model

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'].astype(np.float64), train_data['train_labels'].astype(np.float64)
valid_x, valid_y = valid_data['valid_data'].astype(np.float64), valid_data['valid_labels'].astype(np.float64)

train_x, train_y, valid_x, valid_y = map(
    torch.tensor, (train_x, train_y, valid_x, valid_y)
)

#Question 6.1.1
if True:
    d_in, H, d_out = 1024, 64, 36
    batch_size = 25
    learning_rate = 1e-2
    epochs = 150
    valid_da = {'x': valid_x, 'y': valid_y}
    train_dl = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    modelFC = torch.nn.Sequential(
        torch.nn.Linear(d_in, H),
        torch.nn.Sigmoid(),
        torch.nn.Linear(H, d_out)
    ).double()
    trainedFC = trainModelSGD(train_dl, modelFC, epochs, learning_rate, valid_da)

#Question 6.1.2
if False:
    # Test by overtraining subset
    # numEach = 20
    # start = np.arange(36) * 300
    # end = np.arange(36) * 300 + numEach
    # samps = np.array([np.arange(s,e) for s, e in zip(start,end)]).flatten()
    # sampx = train_x[samps]
    # sampy = train_y[samps]
    # sampx = sampx.reshape((-1,1,32,32))

    batch_size = 25
    learning_rate = 2e-2
    epochs = 25

    train_x = train_x.reshape((-1, 1, 32, 32))
    valid_x = valid_x.reshape((-1,1,32,32))
    valid_da = {'x': valid_x, 'y': valid_y}

    train_dl2 = DataLoader(TensorDataset(train_x, train_y), shuffle=True, batch_size=batch_size)
    modelCNN = torch.nn.Sequential(
        torch.nn.Conv2d(1, 6, 5),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2,2),
        torch.nn.Conv2d(6, 16, 5),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2,2),
        torch.nn.Flatten(),
        torch.nn.Linear(16 * 5 * 5, 120),
        torch.nn.ReLU(),
        torch.nn.Linear(120, 60),
        torch.nn.ReLU(),
        torch.nn.Linear(60, 36)
    ).double()
    trainedCNN = trainModelSGD(train_dl2, modelCNN, epochs, learning_rate, valid_da)
