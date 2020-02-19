import os
import argparse
import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt
from skimage import feature
from sklearn.preprocessing import normalize
from saliency_models import gbvs, ittikoch

def make_header(rangeSize):
    header = "@relation BaseImagens"

    for i in range(rangeSize):
        header += f"\n@attribute h{i} NUMERIC"

    header += "\n@attribute classe {"

    for i in range(1, 9):
        header += f"{i},"
    
    header+= "9}\n@data\n\n"
    
    return header

def saliency(img, model): 
    model = getattr(sys.modules[__name__], model)
    saliency_map = model.compute_saliency(img)
    saliency_map = np.uint8(saliency_map)

    return saliency_map

def threshold(img, range):
    threshold_img = cv2.threshold(img, range, 255, cv2.THRESH_BINARY)[1]
    return threshold_img

def cut(img, shape):
    cut_img = cv2.bitwise_and(img, img, mask = shape)
    return cut_img

def get_histogram(img, range_size):
    hist = cv2.calcHist([img], [0], None, [range_size], [0, range_size])
    hist[0] = 0
    return hist

def normalize_hist(hist):
    hist = hist / np.linalg.norm(hist)
    histString = ''
    for i in hist:
        histString = histString + str(i) + ','
    return histString

def to_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

# def get_LBP(img, numPoints, radius):
#     lbp = feature.local_binary_pattern(img, numPoints, radius, method="uniform")
#     lbp = np.float32(lbp)
#     return lbp

def get_LBP(gray_img):
    imgLBP = np.zeros_like(gray_img)
    neighboor = 3
    for ih in range(0, gray_img.shape[0] - neighboor):
        for iw in range(0, gray_img.shape[1] - neighboor):
            img = gray_img[ih:ih+neighboor, iw:iw+neighboor]
            center = img[1,1]
            img01 = (img >= center)*1.0
            img01_vector = img01.T.flatten()
            img01_vector = np.delete(img01_vector, 4)
            where_img01_vector = np.where(img01_vector)[0]
            if len(where_img01_vector) >= 1:
                num = np.sum(2**where_img01_vector)
            else:
                num = 0
            imgLBP[ih+1, iw+1] = num
    return(imgLBP)

def option1(img, saliency_map, range_size):
    threshold_img = threshold(saliency_map, 100)
    gray_img = to_gray(img)
    cut_img = cut(gray_img, threshold_img)
    hist = get_histogram(cut_img, range_size)
    hist = normalize_hist(hist)
    return hist

def option2(img, saliency_map, range_size):
    threshold_img = threshold(saliency_map, 100)
    gray_img = to_gray(img)
    cut_img = cut(gray_img, threshold_img)
    hist_cut = get_histogram(cut_img, range_size)
    hist_img = get_histogram(gray_img, range_size)
    hist = np.add(hist_cut, hist_img)
    hist = normalize_hist(hist)
    return hist

def option3(img, saliency_map, range_size):
    gray_img = to_gray(img)

    threshold_img1 = threshold(saliency_map, 200)
    cut_img1 = cut(gray_img, threshold_img1)
    hist_cut1 = get_histogram(cut_img1, range_size)

    threshold_img2 = threshold(saliency_map, 150)
    cut_img2 = cut(gray_img, threshold_img2)
    hist_cut2 = get_histogram(cut_img2, range_size)

    threshold_img3 = threshold(saliency_map, 100)
    cut_img3 = cut(gray_img, threshold_img3)
    hist_cut3 = get_histogram(cut_img3, range_size)

    hist = np.add(hist_cut1, hist_cut2)
    hist = np.add(hist, hist_cut3)

    hist = normalize_hist(hist)
    return hist

def option4(img, saliency_map, range_size):
    gray_img = to_gray(img)

    threshold_img1 = threshold(saliency_map, 200)
    cut_img1 = cut(gray_img, threshold_img1)
    hist_cut1 = get_histogram(cut_img1, range_size)

    threshold_img2 = threshold(saliency_map, 150)
    cut_img2 = cut(gray_img, threshold_img2)
    hist_cut2 = get_histogram(cut_img2, range_size)

    threshold_img3 = threshold(saliency_map, 100)
    cut_img3 = cut(gray_img, threshold_img3)
    hist_cut3 = get_histogram(cut_img3, range_size)

    hist_img = get_histogram(gray_img, range_size)

    hist = np.add(hist_cut1, hist_cut2)
    hist = np.add(hist, hist_cut3)
    hist = np.add(hist, hist_img)

    hist = normalize_hist(hist)
    return hist

def option5(img, saliency_map, range_size):
    gray_img = to_gray(img)
    threshold_img = threshold(saliency_map, 125)

    imgLBP = get_LBP(gray_img)

    hist1 = [0] * 256
    hist2 = [0] * 256

    for ih in range(0, imgLBP.shape[0]):
        for iw in range(0, imgLBP.shape[1]):
            if threshold_img[ih][iw] == 0:
                hist1[imgLBP[ih][iw]] += 1
            else: 
                hist2[imgLBP[ih][iw]] += 1
    
    hist = normalize_hist(hist2)
    return hist

def write_histograms_file(path, model, option):
    file_name = f"{model}_option{option}_histogram.arff"
    range_size = 256
    option = 'option' + option

    with open(file_name, 'w') as text:
        text.write(make_header(range_size))
        i = 0
        for d, _, fs in os.walk(path):
            for f in fs:
                imgPath = d + '/' + f
                img = cv2.imread(imgPath)

                if img is not None:
                    saliency_map = saliency(img, model)
                    prepare_func = getattr(sys.modules[__name__], option)
                    hist = prepare_func(img, saliency_map, range_size)
                    hist = hist + ' ' + str(i) + '\n'
                    text.write(hist)
                    print(imgPath + ' ' + str(i))
            i+=1
    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument(
        "-p", "--path", required=True,
        help="path to image folder"
    )
    args.add_argument(
        "-m", "--model", required=True,
        help="saliency model to be used"
    )
    args.add_argument(
        "-o", "--option", required=True,
        help="preparation option"
    )

    args = vars(args.parse_args())
    write_histograms_file(args["path"], args["model"], args["option"])
    