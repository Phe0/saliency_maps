import cv2
import os

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
    path = './MedDB5000/'

    with open('histograma.txt', 'w') as text:
        for d, _, fs in os.walk(path):
            for f in fs:
                imgPath = d + '/' + f
                img = cv2.imread(imgPath)

                if img is not None:
                    img = para_cinza(img)
                    hist = extrair_caracteristica(img) + ' ' + d.split('/')[2] + '\n'
                    text.write(hist)
