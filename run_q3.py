import numpy as np
import scipy.io
from nn import *

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

max_iters = 250
# pick a batch size, learning rate
batch_size = 25
learning_rate = 5e-4
hidden_size = 64

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
initialize_weights(train_x.shape[1],hidden_size,params,'layer1')
initialize_weights(hidden_size,train_y.shape[1],params,'output')

accsTrain = np.zeros((max_iters,1))
accsValid = np.zeros((max_iters,1))
avLossTrain = np.zeros((max_iters,1))
avLossValid = np.zeros((max_iters,1))

# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        # forward
        h1 = forward(xb,params,'layer1')
        probs = forward(h1,params,'output',softmax)

        # loss
        # be sure to add loss and accuracy to epoch totals 
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss += loss
        total_acc += xb.shape[0] * acc
        
        # backward
        delta1 = probs - yb
        delta2 = backwards(delta1,params,'output',linear_deriv)
        backwards(delta2,params,'layer1',sigmoid_deriv)

        # apply gradient
        for k,v in sorted(list(params.items())):
            if 'grad' in k:
                name = k.split('_')[1]
                params[name] -= learning_rate * v

    total_acc /= train_x.shape[0]
    accsTrain[itr] = total_acc
    avLossTrain[itr] = total_loss/train_x.shape[0]

    #loss/acc for validation data

    h1_valid = forward(valid_x,params,'layer1')
    probs_valid = forward(h1_valid,params,'output',softmax)
    loss_valid, acc_valid = compute_loss_and_acc(valid_y, probs_valid)

    accsValid[itr] = acc_valid
    avLossValid[itr] = loss_valid/valid_x.shape[0]

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))


from matplotlib import pyplot as plt
plt.plot(range(max_iters), accsTrain)
plt.plot(range(max_iters), accsValid)
plt.title("Accuracy on Data by Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Training","Validation"])
plt.show()

plt.plot(range(max_iters), avLossTrain)
plt.plot(range(max_iters), avLossValid)
plt.title("Average Loss on Data by Epoch")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.legend(["Training","Validation"])
plt.show()

# run on validation set and report accuracy! should be above 75%
h1 = forward(valid_x,params,'layer1')
probs = forward(h1,params,'output',softmax)
_, valid_acc = compute_loss_and_acc(valid_y, probs)

print('Validation accuracy: ',valid_acc)
if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import copy

def displayWeights(w1):
    indW = np.array([w1[:,ind] for ind in range(w1.shape[1])])
    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols=(8, 8), axes_pad=0.1)

    for ax, im in zip(grid, indW):
        im = im.reshape((32,32))
        ax.imshow(im)

    plt.show()

w1 = copy.copy(params["Wlayer1"])
displayWeights(w1)

#Compare to newly initialized weights
params_temp = {}
initialize_weights(train_x.shape[1],hidden_size,params_temp,'layer1')
initialize_weights(hidden_size,train_y.shape[1],params_temp,'output')

w2 = copy.copy(params_temp["Wlayer1"])
displayWeights(w2)

# Q3.4
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

# compute comfusion matrix here
guess = np.argmax(probs, axis = 1)
correct = np.argmax(valid_y, axis = 1)

for guessInd, corrInd in zip(guess, correct):
    confusion_matrix[guessInd, corrInd] += 1


import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.xlabel("Actual Class")
plt.ylabel("Predicted Class")
plt.show()