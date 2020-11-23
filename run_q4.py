import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):

    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw, cmap='gray')
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()

    #sort bounding boxes by highest lower row
    bboxSorted = sorted(bboxes, key = lambda bbox: bbox[2])
    if len(bboxSorted) == 0:
        continue


    #initiate first line
    row1 = [bboxSorted[0]]
    rows = [row1]

    for bboxInd in range(1, len(bboxSorted)):

        #last height
        lastBottom = bboxSorted[bboxInd - 1][2]
        thisTop = bboxSorted[bboxInd][0]
        if thisTop < lastBottom:
            #overlapping, so tag onto the same row
            rows[-1].append(bboxSorted[bboxInd])
        else:
            #not overlapping, start a new row as a new list
            rows.append([bboxSorted[bboxInd]])

    #sort each row now left to right
    rows = [sorted(row, key = lambda bbox: bbox[1]) for row in rows]

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset

    ims = np.zeros((0,1024))
    for row in rows:
        for box in row:
            #img data
            arr = bw[box[0]:box[2], box[1]:box[3]].astype(np.float64)

            #determine padding size
            sqWidth = max(arr.shape) + 40
            pad = ((sqWidth - np.array((arr.shape))) / 2).astype(int).reshape(2,1)
            pad = np.hstack((pad, pad))

            #pad with background 
            padded = np.pad(arr, pad, constant_values = (1.0,))

            #resize
            sig = (padded.shape[0]/32 -1)/2
            scaled = skimage.transform.resize(padded, (32,32), anti_aliasing=True, anti_aliasing_sigma=sig)

            #Erode (which is really dilation because polarity is flipped)
            erosionSize = 2
            dilated = skimage.morphology.erosion(scaled, skimage.morphology.square(erosionSize))

            ims = np.vstack((ims, dilated.T.flatten()))

    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))

    h1 = forward(ims,params,'layer1')
    probs = forward(h1,params,'output',softmax)
    vals = np.argmax(probs, axis = 1)
    alphaNum = string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)])
    chars = [alphaNum[ind] for ind in vals]

    #if a letter is more than this many pixels from the previous one, its a new word
    spaceDist = bw.shape[1]/19
    message = ''
    charInd = 0
    for bboxRow in rows:
        line = ''
        for bboxInd in range(len(bboxRow)):
            thisBbox = bboxRow[bboxInd]
            if bboxInd:
                lastBbox = bboxRow[bboxInd - 1]
                if (thisBbox[1] - lastBbox[1]) > spaceDist:
                    line += ' '
            line += chars[charInd]
            charInd += 1
        line+='\n'
        message += line      

    print(message)

    #Show the characters fed into the network for comparison and validation
    rowLengths = [len(row) for row in rows]

    gridRows = len(rows)
    gridCols = max(rowLengths)
    from mpl_toolkits.axes_grid1 import ImageGrid
    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols=(gridRows, gridCols), axes_pad=0.1)

    imInd = 0
    for rowInd in range(len(rows)):
        for colInd in range(len(rows[rowInd])):
            thisIm = ims[imInd].reshape(32,32).T
            imInd += 1
            grid[colInd + gridCols * rowInd].imshow(thisIm, cmap='gray')
    plt.show()