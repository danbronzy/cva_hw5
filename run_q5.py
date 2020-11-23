import numpy as np
import scipy.io
from nn import *
from collections import Counter

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
mu = 0.9

batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
initialize_weights(train_x.shape[1],hidden_size,params,'layer1')
initialize_weights(hidden_size,hidden_size,params,'layer2')
initialize_weights(hidden_size,train_x.shape[1],params,'output')

avLossTrain = np.zeros((max_iters,1))
# avLossValid = np.zeros((max_iters,1))

# should look like your previous training loops
for itr in range(max_iters):
    total_loss = 0
    for count, (xb,_) in enumerate(batches):
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        # forward
        h1 = forward(xb,params,'layer1',relu)
        h2 = forward(h1,params,'layer2',relu)
        probs = forward(h2,params,'output',sigmoid)

        # loss
        # be sure to add loss and accuracy to epoch totals 
        loss = np.sum((xb - probs)**2)
        total_loss += loss
        
        # backward
        delta1 = 2*(probs - xb) * probs * (1-probs)
        delta2 = backwards(delta1,params,'output',linear_deriv)
        delta3 = backwards(delta2,params,'layer2',relu_deriv)
        backwards(delta3,params,'layer1',relu_deriv)

        # apply gradient
        for k,v in sorted(list(params.items())):
            if 'grad' in k:
                name = k.split('_')[1]
                params['m_'+name] = mu * params['m_'+name] - learning_rate * v
                params[name] += params['m_'+name]

    avLossTrain[itr] = total_loss/train_x.shape[0]

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9

import matplotlib.pyplot as plt

plt.plot(range(max_iters), avLossTrain)
plt.xlabel("Epoch")
plt.ylabel("Average Sample Loss")
plt.show()
# Q5.3.1

# visualize some results
testVals = [11,20,310,333,1650,1690,2467,2456,3209,3276]
ims = np.array([valid_x[row, :] for row in testVals])
res = np.zeros((0,1024))
for im in ims:
    #eval on network
    h1 = forward(im,params,'layer1',relu)
    h2 = forward(h1,params,'layer2',relu)
    probs = forward(h2,params,'output',sigmoid)
    res = np.vstack((res,probs))


for real, recon in zip(ims, res):
    ax = plt.subplot(121)
    ax.imshow(real.reshape((32,32)).T, cmap='gray')
    ax.set_title("Input Image")
    ax.axis('off')
    ax2 = plt.subplot(122)
    ax2.imshow(recon.reshape((32,32)).T, cmap='gray')
    ax2.set_title("Reconstructed Image")
    ax2.axis('off')
    plt.show()


# Q5.3.2
from skimage.measure import compare_psnr as psnr
# evaluate PSNR

#eval network on all images
psnrs = np.zeros(valid_x.shape[0])
for ind in range(valid_x.shape[0]):
    #eval on network
    real = valid_x[ind, :]

    h1 = forward(real,params,'layer1',relu)
    h2 = forward(h1,params,'layer2',relu)
    probs = forward(h2,params,'output',sigmoid)

    v = psnr(real, probs)
    psnrs[ind] = v

np.mean(psnrs)
