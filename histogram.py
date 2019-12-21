from skimage import feature
import numpy as np
import cv2
import os

def make_header(rangeSize):
    header = "@relation BaseImagens"

    for i in range(rangeSize):
        header += f"\n@attribute h{i} NUMERIC"

    header += "\n@attribute classe {"

    for i in range(1, 32):
        header += f"{i},"
    
    header+= "32}\n@data\n\n"
    
    return header

def get_gray_scale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # get gray scale of the image
    return gray

def get_histogram(img):
    print(img.dtype)
    print(img)
    hist = cv2.calcHist([img], [0], None, [27], [0, 26]) # get histogram

    cv2.normalize(hist, hist) # normalizes the histogram
    histString = ''

    for i in hist:
        histString = histString + str(i[0]) + ',' # get histogram string
    print(histString)
    return histString

def get_LBP(img, numPoints, radius):
    lbp = feature.local_binary_pattern(img, numPoints, radius, method="uniform")
    lbp = np.float32(lbp)
    return lbp


def write_histograms_file(path): # write file with histograms
    with open('histograms.arff', 'w') as text:
        text.write(make_header(27))
        i = 0
        for d, _, fs in os.walk(path): 
            for f in fs:
                imgPath = d + '/' + f
                img = cv2.imread(imgPath)

                if img is not None:
                    img = get_gray_scale(img)
                    img = get_LBP(img, 24, 8)
                    hist = get_histogram(img) + ' ' + str(i) + '\n'
                    text.write(hist)
            i+=1

path = './MedDB5000/'

write_histograms_file(path)

