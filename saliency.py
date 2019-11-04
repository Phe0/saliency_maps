import cv2
import argparse

args = argparse.ArgumentParser()
args.add_argument(
        "-i", "--image", required=False,
        help="path to input image")
args = vars(args.parse_args())

image = cv2.imread(args["image"])

#saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
saliency = cv2.saliency.StaticSaliencyFineGrained_create()
success, saliency_map = saliency.computeSaliency(image)
saliency_map = (saliency_map*255).astype("uint8")
#cv2.imshow("Image Original", image)
#cv2.imshow("Output", saliency_map)

cv2.imshow("Saliency Map", saliency_map)
thresh_map = cv2.threshold(saliency_map.astype("uint8"), 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Thresh Map", thresh_map)
heat_map = cv2.applyColorMap(thresh_map, cv2.COLORMAP_RAINBOW)
cv2.imshow("Color Map", heat_map)
new_name = "./outputs/3.jpg"
cv2.imwrite(new_name, heat_map)
cv2.waitKey(0)
