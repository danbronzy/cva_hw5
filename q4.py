import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
# insert processing in here
# one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
# this can be 10 to 15 lines of code using skimage functions

    openSize = 10
    closeSize = 3
    erosionSize = 3
    bboxMinArea = 350

    sig_est = skimage.restoration.estimate_sigma(image, average_sigmas=True, multichannel=True)
    smoothed = skimage.filters.gaussian(image, sigma = sig_est, multichannel=True)
    # denoised = skimage.restoration.denoise_tv_chambolle(image, multichannel=True)


    grayscale = skimage.color.rgb2gray(smoothed)

    thresh = skimage.filters.threshold_otsu(grayscale)
    binary = grayscale > thresh

    #reduce background noise, connect foreground
    closed = skimage.morphology.binary_closing(binary, skimage.morphology.square(closeSize)) 
    opened = skimage.morphology.binary_opening(closed, skimage.morphology.square(openSize))
    dilated = skimage.morphology.binary_erosion(opened, skimage.morphology.square(erosionSize))
    labels = skimage.measure.label(dilated, background=1.0)

    rps =  skimage.measure.regionprops(labels)

    bboxes = []
    for rp in rps:
        if rp.bbox_area >= bboxMinArea:

            #Compare the centroid of this region to the rest of the regions.
            #If the centroid lies inside another region, and this region is smaller,
            #don't add it to the list
            adding = True
            for op in rps:
                if rp.bbox == op.bbox:
                    #same region
                    continue
                #comparing different regions
                minr, minc, maxr, maxc = op.bbox
                cenr, cenc = rp.centroid

                #if its inside another bounding box
                if (minr < cenr < maxr) and (minc < cenc < maxc):

                    #and its the smaller of the bounding boxes
                    if rp.bbox_area < op.bbox_area:
                        #don't add it
                        adding = False
            if adding:
                bboxes.append(rp.bbox)

    bw = dilated
    return np.array(bboxes), bw