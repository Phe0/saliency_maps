import cv2
import os

def make_header():
    header = "@relation BaseImagens"

    for i in range(256):
        header += f"\n@attribute h{i} NUMERIC"

    header += "\n@attribute classe NUMERIC\n@data\n\n"
    
    return header

def get_gray_scale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # get gray scale of the image
    return gray

def get_histogram(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]) # get histogram

    cv2.normalize(hist, hist) # normalizes the histogram
    histString = ''

    for i in hist:
        histString = histString + str(i[0]) + ',' # get histogram string
    return histString

def write_histograms_file(path): # write file with histograms
    with open('histograms.arff', 'w') as text:
        text.write(make_header())
        i = 0
        for d, _, fs in os.walk(path): 
            for f in fs:
                imgPath = d + '/' + f
                img = cv2.imread(imgPath)

                if img is not None:
                    img = get_gray_scale(img)
                    hist = get_histogram(img) + ' ' + str(i) + '\n'
                    text.write(hist)
            i+=1

path = './MedDB5000/'
write_histograms_file(path)

