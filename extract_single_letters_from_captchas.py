
import os
import os.path
import cv2
import glob
import imutils
import numpy as np
import solving_captchas_code_examples.helpers

CAPTCHA_IMAGE_FOLDER = "../train"
OUTPUT_FOLDER = "extracted_letter_images"


# Get a list of all the captcha images we need to process
captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))

def doOverlap(l1x , l1y , r1x ,r1y, l2x , l2y, r2x ,r2y):
    # print(str(l1x) + " " + str(l1y) + " " + str(r1x) + " " + str(r1y) + " " + str(l2x) + " " + str(l2y) + " " + str(r2x) + " " + str(r2y))
    # If one rectangle is on left side of other
    if (l1x > r2x or l2x > r1x):
        return False

    # If one rectangle is above other
    if (l1y > r2y or l2y > r1y):
        return False

    return True

counts = {}
total_count = 0
more_less_count = 0
# loop over the image paths
for (i, captcha_image_file) in enumerate(captcha_image_files):
    print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files)))

    # Since the filename contains the captcha text (i.e. "2A2X.png" has the text "2A2X"),
    # grab the base filename as the text
    filename = os.path.basename(captcha_image_file)
    captcha_correct_text = os.path.splitext(filename)[0]

    # Load the image and convert it to grayscale
    image = cv2.imread(captcha_image_file)
    pixelColor = image[0, 0]
    # print(pixel)
    background = np.zeros(image.shape, np.uint8)
    background[:, :] = pixelColor

    blackBackGround = 255 + image - background

    kern = np.ones((6, 6), np.uint8)

    blackBackGround = cv2.dilate(blackBackGround, kern, iterations=2)
    blackBackGround = cv2.erode(blackBackGround, kern, iterations=1)
    # cv2.imshow("Bef",image)
    # kern = np.ones((5, 8), np.uint8)
    #
    # blackBackGround = cv2.dilate(blackBackGround, kern, iterations=3)
    # blackBackGround = cv2.erode(blackBackGround, kern, iterations=1)
    # cv2.imshow("After diler", blackBackGround)
    # cv2.waitKey()
    gray = cv2.cvtColor(blackBackGround, cv2.COLOR_BGR2GRAY)

    gray = cv2.copyMakeBorder(gray, 20, 20, 20, 20, cv2.BORDER_REPLICATE)

    finalGray = gray

    ret, gray = cv2.threshold(gray, 251, 255, cv2.THRESH_BINARY_INV)
    ret, finalGray = cv2.threshold(finalGray, 251, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow("After Threshold", gray)

    _, contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # print("Number of Contours found " + filename + " = " + str(len(contours)))

    blank = np.ones(image.shape)

    # cv2.waitKey(0)

    letter_image_regions = []

    for contour in contours:
        # Get the rectangle that contains the contour
        (x, y, w, h) = cv2.boundingRect(contour)

        if (w < 10):
            continue
        if (h < 40):
            continue

        if (w * h < 200):
            continue

        if w / h > 1.5:
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
            cv2.rectangle(blank, (x, y), (x + half_width, y + h), (0, 0, 0), 2)
            cv2.rectangle(blank, (x + half_width, y), (x + half_width, y + h), (0, 0, 0), 2)
            continue
        add = True
        cv2.rectangle(blank, (x, y), (x + w, y + h), (0, 0, 0), 2)
        for index, image_regions in enumerate(letter_image_regions):
            x1, y1, w1, h1 = image_regions
            rvalue = doOverlap(x, y, x + w, y + h, x1, y1, x1 + w1, y1 + h1)
            if (rvalue):
                # print("do overlap")
                if (w1 * h1 < w * h):
                    letter_image_regions.remove(image_regions)
                    add = True
                    # print("removed "+str(image_regions))
                else:
                    add = False
        # print()
        # print()
        if (add):
            letter_image_regions.append((x, y, w, h))
    # cv2.imshow("contours" ,blank)

    # if len(letter_image_regions) != len(captcha_correct_text):
    #     print("Wrong letters " + str(filename))
    #     more_less_count = more_less_count + 1
    # continue

    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
    ttt = 0

    # print("Length " + str(len(letter_image_regions)))
    output = cv2.merge([image] * 3)
    for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_correct_text):
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box

        # Extract the letter from the original image with a 2-pixel margin around the edge
        letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]

        # Get the folder to save the image in
        save_path = os.path.join(OUTPUT_FOLDER, letter_text)

        # if the output directory does not exist, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # write the letter image to a file
        count = counts.get(letter_text, 1)
        p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        cv2.imwrite(p, letter_image)

        # increment the count for the current key
        counts[letter_text] = count + 1

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #
    # # Add some extra padding around the image
    # gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
    #
    # # threshold the image (convert it to pure black and white)
    # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    #
    # # find the contours (continuous blobs of pixels) the image
    # contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    # # Hack for compatibility with different OpenCV versions
    # contours = contours[0] if imutils.is_cv2() else contours[1]
    #
    # letter_image_regions = []
    #
    # # Now we can loop through each of the four contours and extract the letter
    # # inside of each one
    # for contour in contours:
    #     # Get the rectangle that contains the contour
    #     (x, y, w, h) = cv2.boundingRect(contour)
    #
    #     # Compare the width and height of the contour to detect letters that
    #     # are conjoined into one chunk
    #     if w / h > 1.25:
    #         # This contour is too wide to be a single letter!
    #         # Split it in half into two letter regions!
    #         half_width = int(w / 2)
    #         letter_image_regions.append((x, y, half_width, h))
    #         letter_image_regions.append((x + half_width, y, half_width, h))
    #     else:
    #         # This is a normal letter by itself
    #         letter_image_regions.append((x, y, w, h))
    #
    # # If we found more or less than 4 letters in the captcha, our letter extraction
    # # didn't work correcly. Skip the image instead of saving bad training data!
    # if len(letter_image_regions) != 4:
    #     continue
    #
    # # Sort the detected letter images based on the x coordinate to make sure
    # # we are processing them from left-to-right so we match the right image
    # # with the right letter
    # letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
    #
    # # Save out each letter as a single image
    # for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_correct_text):
    #     # Grab the coordinates of the letter in the image
    #     x, y, w, h = letter_bounding_box
    #
    #     # Extract the letter from the original image with a 2-pixel margin around the edge
    #     letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]
    #
    #     # Get the folder to save the image in
    #     save_path = os.path.join(OUTPUT_FOLDER, letter_text)
    #
    #     # if the output directory does not exist, create it
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #
    #     # write the letter image to a file
    #     count = counts.get(letter_text, 1)
    #     p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
    #     cv2.imwrite(p, letter_image)
    #
    #     # increment the count for the current key
    #     counts[letter_text] = count + 1
