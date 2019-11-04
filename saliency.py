import cv2
import time
import argparse
import numpy as np
from matplotlib import pyplot as plt
from gbvs import gbvs

def run(image):
    params = gbvs.setupParams()

    mask = gbvs.run(image/255, params) * 255
    mask = np.uint8(mask)
    cv2.imshow("Mask GBVS", mask)

    threshed = cv2.threshold(mask, 75, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("Gray scale", threshed)
    colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

    masked = cv2.bitwise_and(colored, colored, mask = threshed)
    addition = cv2.addWeighted(image, 1.0, masked, 0.6, 0)
    return addition

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument(
            "-i", "--image", required=True,
            help="path to input image")
    args.add_argument(
            "-o", "--output", required=True,
            help="path to save the output image")
    args = vars(args.parse_args())
    image = cv2.imread(args["image"])
    #out_name = "./outputs/3.jpg"
    image = run(image)
    cv2.imshow("Colored map", image)
    #cv2.imwrite(out_name, image)
    cv2.imwrite(args["output"], image)
    plt.show()
    cv2.waitKey(0)
