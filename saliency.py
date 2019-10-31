import cv2
import time
import gbvs.gbvs as gbvs
import numpy as np
from matplotlib import pyplot as plt

def run(path):
    params = gbvs.setupParams()
    image = cv2.imread(path)
    image = image / 255.0

    mask = gbvs.run(image, params) * 255.0
    mask = np.uint8(mask)

    (T, thresh) = cv2.threshold(mask, 75, 255, cv2.THRESH_BINARY)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    

    masked = cv2.bitwise_and(colored, colored, mask = thresh)
    addition = cv2.addWeighted(image, 1.0, masked, 0.1, 0, dtype=cv2.CV_32F)
    return addition

if __name__ == '__main__':

    image = run('./images/1.jpg')
    oname = "./outputs/1.jpg"
    cv2.imwrite(oname, image)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()