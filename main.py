import cv2
import numpy as np
import sys

# constants
IMAGE_COUNT = 7
MAX_WIDTH = 700

thresh = 0.60
num_thresh = 0.80
connectivity = 4
template = cv2.imread('train/eyvel_win.PNG')

# reshapes the output image
def reshape(img):
    if img.shape[1] > MAX_WIDTH:
        s = MAX_WIDTH / img.shape[1]
        img = cv2.resize(img, dsize=None, fx=s, fy=s)

    return img

for i in range(1, IMAGE_COUNT + 1):
    # read in query image
    query_img = cv2.imread('screen/screen' + str(i) + '.PNG')
    bgr_image_output = query_img.copy()

    # score the image to find the object
    scores = cv2.matchTemplate(query_img, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(scores)
    max_score = -1

    if(max_val > max_score):
        max_score = max_val

    # set the location of the matched object
    topLeft = max_loc
    bottomRight = (topLeft[0] + (template.shape[1]), topLeft[1] + (template.shape[0]))

    # if match exceeds threshold, we have a good match
    if(max_val > thresh):
        cv2.rectangle(bgr_image_output, topLeft, bottomRight, (0, 0, 255), 2)

        for i in range(0, 10):
            # find numbers within the forecast
            num = cv2.imread('nums/r' + str(i) + '.PNG')
            scores = cv2.matchTemplate(query_img, num, cv2.TM_CCOEFF_NORMED)

            # Threshold the image and extract centroids
            thresholded = (scores > num_thresh).astype(np.uint8)
            statInfo = cv2.connectedComponentsWithStats(thresholded, connectivity, cv2.CV_32S)
            centroids = statInfo[3]

            # Throw out the background centroid
            centroids = centroids[1:]

            # Draw rectangles around each letter
            for centroid in centroids:
                color = (255, 255, 255)
                cv2.rectangle(bgr_image_output, (int(centroid[0]), int(centroid[1]), int(num.shape[1]), int(num.shape[0])),
                          color, thickness=4)
                

    bgr_image_output = reshape(bgr_image_output)
    cv2.imshow("Output", bgr_image_output)
    cv2.waitKey(0)
