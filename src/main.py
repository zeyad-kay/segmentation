import cv2
import numpy as np
from kmeans import KMeans
from luv import luv_rgb, rgb_luv
inputImage = cv2.imread("./images/face.jpg", cv2.IMREAD_COLOR)
print("rgb_luv")
luv = rgb_luv(inputImage)
pixel_vals = luv.reshape((-1,3))
# Convert to float type
pixel_vals = np.float32(pixel_vals)
print("kmeans")
m = KMeans(K=3,max_iter=10).fit(pixel_vals)

# convert data into 8-bit values
centers = np.uint8(m.centroids)
segmented_data = centers[m.labels.flatten()]
# reshape data into the original image dimensions
segmented_luv = segmented_data.reshape((inputImage.shape))
print("luv_rgb")
segmented_image = luv_rgb(segmented_luv)
cv2.imshow("original",inputImage)
cv2.imshow("segmented",segmented_image)
cv2.waitKey(0)