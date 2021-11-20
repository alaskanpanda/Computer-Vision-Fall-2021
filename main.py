import cv2
import numpy as np
import sys

# constants
IMAGE_COUNT = 7
MAX_WIDTH = 700

thresh = 0.60
num_thresh = 0.85
connectivity = 4
template = cv2.imread('train/eyvel_win.PNG')

# reshapes the output image
def reshape(img):
    if img.shape[1] > MAX_WIDTH:
        s = MAX_WIDTH / img.shape[1]
        img = cv2.resize(img, dsize=None, fx=s, fy=s)

    return img

# takes numbers from template matching step, maps them to 2 digit nums
def getValuesFromNumbers(inputs):
    # inputs = np.array[[x, y, value], [x, y, value], ...]
    outputs = []
    byY = []
    adjacentTolorance = 100
    yTolerance = 10

    # group digits by y value
    for num in inputs:
        x = num[0]
        y = num[1]

        if len(byY) == 0:
            byY.append([num])
        else:
            added = False
            for i in range(len(byY)):
                if abs(byY[i][0][1] - y) <= yTolerance:
                    added = True
                    byY[i].append(num)

            if not added:
                byY.append([num])

    # combine adjacent digits
    for i in range(len(byY)):
        for j in range(len(byY[i])):
            otherFound = False

            if (byY[i][j][2] < 0):
                continue

            for k in range(j + 1, len(byY[i])):
                if abs(byY[i][j][0] - byY[i][k][0]) <= adjacentTolorance:
                    otherFound = True

                    if byY[i][j][0] > byY[i][k][0]:
                        outputs.append([byY[i][j][0], byY[i][j][1], byY[i][k][2] * 10 + byY[i][j][2]])
                    else:
                        outputs.append([byY[i][k][0], byY[i][k][1], byY[i][j][2] * 10 + byY[i][k][2]])

                    byY[i][k][2] = -1
                    break

            if not otherFound:
                outputs.append(byY[i][j])

    # return numbers
    return np.array(outputs)


# list of numbers found in the image with their (x, y) coordinates
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
    forecast_nums = []

    # if match exceeds threshold, we have a good match
    if(max_val > thresh):
        cv2.rectangle(bgr_image_output, topLeft, bottomRight, (0, 0, 255), 2)

        ## NOTE: might be a good idea to template match the rectangle, not the whole image ##
        ##       should improve performance to work in real-time                           ##

        for i in range(0, 10):
            # find numbers within the forecast
            num = cv2.imread('nums/r' + str(i) + '.PNG')

            gray_template = cv2.cvtColor(num, cv2.COLOR_BGR2GRAY)
            gray_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)

            scores = cv2.matchTemplate(gray_img, gray_template, cv2.TM_CCOEFF_NORMED)

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
                forecast_nums.append([int(centroid[0]), int(centroid[1]), i])

        dash = cv2.imread('nums/na.PNG')
        scores = cv2.matchTemplate(query_img, dash, cv2.TM_CCOEFF_NORMED)

        # Threshold the image and extract centroids
        thresholded = (scores > num_thresh).astype(np.uint8)
        statInfo = cv2.connectedComponentsWithStats(thresholded, connectivity, cv2.CV_32S)
        centroids = statInfo[3]
        # Throw out the background centroid
        centroids = centroids[1:]

        # Draw rectangles around each letter
        for centroid in centroids:
            color = (255, 255, 255)
            cv2.rectangle(bgr_image_output, (int(centroid[0]), int(centroid[1]), int(dash.shape[1]), int(dash.shape[0])),
                          color, thickness=4)

    organized_nums = getValuesFromNumbers(forecast_nums)
    print("~~~~~~~~~~~~~~~~~~~~~~~")
    for thing in organized_nums:
        print(thing)


    bgr_image_output = reshape(bgr_image_output)
    cv2.imshow("Output", bgr_image_output)
    cv2.waitKey(0)
