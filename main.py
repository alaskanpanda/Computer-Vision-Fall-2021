import cv2
import numpy as np
import sys

# constants
MAX_WIDTH = 700
OUTPUT_FRAME_SIZE = 400

orb_thresh = 0.70
num_thresh = 0.60
min_matches = 10
connectivity = 4
mp4_file = "gameplay.mp4"
template = cv2.imread('train/eyvel_win.PNG')
video_capture = cv2.VideoCapture(mp4_file)     # Open video capture object

if(mp4_file == "gameplay.mp4"):
    orb_thresh = 0.70
    num_thresh = 0.80
    min_matches = 40
elif(mp4_file == "gameplay_phone.mp4"):
    orb_thresh = 0.70
    num_thresh = 0.60
    min_matches = 10
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

found_prev_frame = False
got_image, bgr_image = video_capture.read()       # Make sure we can read video
if not got_image:
    print("Cannot read video source")
    sys.exit()

# reshapes the output image
def reshape(img):
    if img.shape[1] > MAX_WIDTH:
        s = MAX_WIDTH / img.shape[1]
        img = cv2.resize(img, dsize=None, fx=s, fy=s)

    return img

# reshape input image to match template image size
def reshape_for_template(img):
    return cv2.resize(img, (730, 1870))

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

# given a numpy array of 2 digit numbers, sort them by order (up/down then left/right)
def getStats(input):
    sorted_forecast_nums = sorted(input, key=lambda element: (element[1], element[0]))
    stats = []

    i = 0
    while i < len(sorted_forecast_nums):
        if(sorted_forecast_nums[i][0] > sorted_forecast_nums[i+1][0]):
            temp = sorted_forecast_nums[i]
            sorted_forecast_nums[i] = sorted_forecast_nums[i+1]
            sorted_forecast_nums[i+1] = temp

        stats.append([sorted_forecast_nums[i][2], sorted_forecast_nums[i+1][2]])
        i += 2

    return stats

# calculate the odds of a successful player-phase attack, given the stat readout
def calcChances(stats):
    killChance, deathChance = 0, 0

    # save stats to be human readable
    enemyHP, playerHP = stats[1][0], stats[1][1]
    enemyATK, playerATK = stats[2][0], stats[2][1]
    enemyDEF, playerDEF = stats[3][0], stats[3][1]
    enemyHIT, playerHIT = stats[4][0] * .01, stats[4][1] * .01     # convert hit/crit rates to
    enemyCRIT, playerCRIT = stats[5][0] * .01, stats[5][1] * .01   # decimal for mulitplication
    enemyAS, playerAS = stats[6][0], stats[6][1]

    # player attack
    if(enemyHP <= (playerATK - enemyDEF)): killChance = playerHIT * 100                           # chance of hit (no crit)
    elif(enemyHP <= ((playerATK * 2) - enemyDEF)): killChance = (playerHIT * playerCRIT) * 100    # chance of hit (with crit)

    # enemy attack
    if(playerHP < (enemyATK - playerDEF)): deathChance = ((1 - killChance) * enemyHIT) * 100                           # chance of hit (no crit)
    elif(playerHP < ((enemyATK * 2) - playerDEF)): deathChance = ((1 - killChance) * (enemyHIT * enemyCRIT)) * 100     # chance of hit (no crit)

    # follow-up attack
    if(playerAS - enemyAS >= 4):
        # do player attack
        if(enemyHP < ((playerATK - enemyDEF) * 2)): killChance = (playerHIT * playerHIT) * 100                                # chance of doubling (no crit)
        elif(enemyHP < ((playerATK * 2 - enemyDEF) + (playerATK - enemyDEF))): ((playerHIT * playerCRIT) * playerHIT) * 100   # chance of doubling (1 crit)
        elif(enemyHP < ((playerATK * 2 - enemyDEF) * 2)): ((playerHIT * playerCRIT) * (playerHIT * playerCRIT)) * 100         # chance of doubling (2 crits)
    elif(enemyAS - playerAS >= 4):
        # do enemy attack
        if(playerHP < ((enemyATK - playerDEF) * 2)): killChance = (enemyHIT * enemyHIT) * 100                               # chance of doubling (no crit)
        elif(playerHP < ((enemyATK * 2 - playerDEF) + (enemyATK - playerDEF))): ((enemyHIT * enemyCRIT) * enemyHIT) * 100   # chance of doubling (1 crit)
        elif(playerHP < (enemyATK * 2 - playerDEF)): ((enemyHIT * enemyCRIT) * (enemyHIT * enemyCRIT)) * 100                # chance of doubling (2 crits)


    # output results
    chance_img = np.ones((int(OUTPUT_FRAME_SIZE * .15), OUTPUT_FRAME_SIZE))
    killChance = str(round(killChance, 2))
    deathChance = str(round(deathChance, 2))
    kill_msg = "Chance of killing the enemy: " + str(killChance) + "%"
    death_msg = "Chance of being killed: " + str(deathChance) + "%"
    cv2.putText(chance_img, kill_msg, org=(10, 25), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0,255,255))
    cv2.putText(chance_img, death_msg,org=(10, 45), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0,255,255))
    cv2.imshow("Chance", chance_img)


while True:

    got_image, bgr_image = video_capture.read()
    if not got_image:
        break  # End of video; exit the while loop

    # read in query image
    query_img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    bgr_image_output = bgr_image

    # do feature matching for video
    orb = cv2.ORB_create()
    trainKeypoints, trainDesc = orb.detectAndCompute(template, None)
    queryKeypoints, queryDesc = orb.detectAndCompute(query_img, None)

    # create and use matcher
    matcher = cv2.BFMatcher.create(cv2.NORM_L2)
    matches = matcher.knnMatch(queryDesc, trainDesc, k=2)
    good = []
    for m, n in matches:
        if m.distance < orb_thresh * n.distance:
            good.append(m)
    matches = good

    bgr_matches = cv2.drawMatches(
        img1=query_img, keypoints1=queryKeypoints,
        img2=template, keypoints2=trainKeypoints,
        matches1to2=matches, matchesMask=None, outImg=None
    )

    A_train_query = None
    warped_image = None
    if len(matches) >= min_matches:

        # Estimate affine transformation from training to query image points.
        # Use the "least median of squares" method for robustness. It also detects outliers.
        # Outliers are those points that have a large error relative to the median of errors.
        src_pts = np.float32([trainKeypoints[m.trainIdx].pt for m in matches]).reshape(
            -1, 1, 2)
        dst_pts = np.float32([queryKeypoints[m.queryIdx].pt for m in matches]).reshape(
            -1, 1, 2)
        A_train_query, inliers = cv2.estimateAffine2D(
            dst_pts, src_pts,
            method=cv2.LMEDS)

    # Apply the affine warp to warp the training image to the query image.
    if A_train_query is not None and len(inliers) >= 3:
        # Object detected! Warp the training image to the query image and blend the images.
        warped_image = cv2.warpAffine(
            src=query_img, M=A_train_query,
            dsize=(template.shape[1], template.shape[0]))


    # if match exceeds threshold, we found the forecast
    if warped_image is not None and not found_prev_frame:
        found_prev_frame = True
        forecast_nums = []

        # extract the part of the image containing the battle forecast
        subset_img = warped_image
        subset_img = reshape_for_template(subset_img)  # reshape battle forecast to match template size

        for i in range(0, 10):
            # find numbers within the forecast
            num = cv2.imread('nums/r' + str(i) + '.PNG')

            gray_num = cv2.cvtColor(num, cv2.COLOR_BGR2GRAY)
            subset_copy = subset_img.copy()
            scores = cv2.matchTemplate(subset_img, gray_num, cv2.TM_CCOEFF_NORMED)

            # Threshold the image and extract centroids
            thresholded = (scores > num_thresh).astype(np.uint8)
            statInfo = cv2.connectedComponentsWithStats(thresholded, connectivity, cv2.CV_32S)
            centroids = statInfo[3]

            # Throw out the background centroid
            centroids = centroids[1:]
            warped_image_copy = subset_img.copy()
            warped_image_copy = cv2.cvtColor(warped_image_copy, cv2.COLOR_GRAY2RGB)

            # Each centroid represents the location of a number, so save it as a forecast num
            for centroid in centroids:
                forecast_nums.append([int(centroid[0]), int(centroid[1]), i])


        dash = cv2.imread('nums/na.PNG')
        dash = cv2.cvtColor(dash, cv2.COLOR_BGR2GRAY)
        scores = cv2.matchTemplate(subset_img, dash, cv2.TM_CCOEFF_NORMED)

        # Threshold the image and extract centroids
        thresholded = (scores > num_thresh).astype(np.uint8)
        statInfo = cv2.connectedComponentsWithStats(thresholded, connectivity, cv2.CV_32S)
        centroids = statInfo[3]

        # Throw out the background centroid
        centroids = centroids[1:]
        thresh_copy = thresholded.copy()

        # Draw rectangles around each letter
        for centroid in centroids:
            forecast_nums.append([int(centroid[0]), int(centroid[1]), 0])

        # convert forecast nums to 2 digit numbers and organize them as stats
        forecast_nums = getValuesFromNumbers(forecast_nums)
        stats = getStats(forecast_nums)

        # calculate chances of killing/being killed
        calcChances(stats)

    # don't repeat calculations that are already shown
    elif warped_image is not None:
        found_prev_frame = True

    # no battle forecast found, display nothing
    else:
        # set display to be blank
        chance_img = np.ones((int(OUTPUT_FRAME_SIZE * .15), OUTPUT_FRAME_SIZE))
        cv2.imshow("Chance", chance_img)
        found_prev_frame = False

    # output image to screen
    bgr_image_output = reshape(bgr_image_output)
    cv2.imshow("Output", bgr_image_output)
    if cv2.waitKey(1) == ord('q'):
        break
