import cv2
import time
import argparse
import numpy as np
from matplotlib import pyplot as plt
from gbvs import gbvs

def run(image):
    params = gbvs.setupParams()
    image = image / 255.0

    mask = gbvs.run(image, params) * 255.0
    mask = np.uint8(mask)
    plt.imshow("mask", mask)

    (T, thresh) = cv2.threshold(mask, 75, 255, cv2.THRESH_BINARY)
    plt.imshow("T", T)
    plt.imshow("tresh", thresh)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    plt.imshow("mask", mask)
    colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    

    masked = cv2.bitwise_and(colored, colored, mask = thresh)
    addition = cv2.addWeighted(image, 1.0, masked, 0.1, 0, dtype=cv2.CV_32F)
    return addition

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument(
            "-i", "--image", required=True,
            help="path to input image")
    args = vars(args.parse_args())
    image = cv2.imread(args["image"])
    out_name = "./outputs/3.jpg"
    image = run(image)
    cv2.imwrite(out_name, image)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.imshow(image)
    plt.show()
