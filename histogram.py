import cv2
import numpy as np 
import matplotlib.pyplot as plt
import os
from statistics import mode

def para_cinza(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def extrair_caracteristica(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    histString = ''

    for i in hist:
        histString = histString + str(i[0]) + ' '
    return histString

if __name__ == '__main__':
    path = './images'
    files = []

    text = open('histograma.txt', 'a+')

    for r, d, f in os.walk(path):
        for dir in d:
            for r1, d1, f1 in os.walk(path + '/' + dir):
                for file in f1:
                    if '.jpg' in file:
                        imgPath = path + '/' + dir + '/' + file
                        img = cv2.imread(imgPath)
                        img = para_cinza(img)
                        hist = extrair_caracteristica(img) + ' ' + dir + '\n'
                        text.write(hist)

