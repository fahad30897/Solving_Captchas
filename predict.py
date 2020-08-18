
from keras.models import load_model
from helpers import resize_to_fit
import imutils
import numpy as np
import cv2
import pickle
import os



# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# INPUT CONVENTION
# filenames: a list of strings containing filenames of images

# OUTPUT CONVENTION
# The method must return a numpy array (not numpy matrix or scipy matrix) and a list of strings.
# Make sure that the length of the array and the list is the same as the number of filenames that 
# were given. The evaluation code may give unexpected results if this convention is not followed.

MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"
# Load up the model labels (so we can translate model predictions to actual letters)
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# Load the trained neural network
model = load_model(MODEL_FILENAME)


def doOverlap(l1x , l1y , r1x ,r1y, l2x , l2y, r2x ,r2y):
    # print(str(l1x) + " " + str(l1y) + " " + str(r1x) + " " + str(r1y) + " " + str(l2x) + " " + str(l2y) + " " + str(r2x) + " " + str(r2y))
    # If one rectangle is on left side of other
    if (l1x > r2x or l2x > r1x):
        return False

    # If one rectangle is above other
    if (l1y > r2y or l2y > r1y):
        return False

    return True

def solve(filenames):

    more_less_count = 0
    total_count = 0
    numchars = []
    codes = []
    for (i, captcha_image_file) in enumerate(filenames):
        # filename = os.path.basename("train/AAA.png")#(captcha_image_file)
        # print(captcha_image_file)
        filename = os.path.basename(captcha_image_file)

        # captcha_correct_text = os.path.splitext("AAA")[0]#(filename)
        # captcha_correct_text = os.path.splitext(filename)[0]

        # Load the image and convert it to grayscale
        # image = cv2.imread("train/AAA.png")#(captcha_image_file)
        image = cv2.imread(captcha_image_file)
        # cv2.waitKey(0)

        # print("Image file {} -> {}", captcha_image_file , image)
        # if(image == None):
        #     continue

        pixelColor = image[0, 0]
        # print(pixel)
        background = np.zeros(image.shape, np.uint8)
        background[:, :] = pixelColor

        blackBackGround = 255 + image - background

        kern = np.ones((6, 6), np.uint8)

        blackBackGround = cv2.dilate(blackBackGround, kern, iterations=2)
        blackBackGround = cv2.erode(blackBackGround, kern, iterations=1)
        # kern = np.ones((5, 8), np.uint8)
        #
        # blackBackGround = cv2.dilate(blackBackGround, kern, iterations=3)
        # blackBackGround = cv2.erode(blackBackGround, kern, iterations=1)
        # cv2.imshow("After diler", blackBackGround)

        gray = cv2.cvtColor(blackBackGround, cv2.COLOR_BGR2GRAY)

        gray = cv2.copyMakeBorder(gray, 20, 20, 20, 20, cv2.BORDER_REPLICATE)

        finalGray = gray

        ret, gray = cv2.threshold(gray, 251, 255, cv2.THRESH_BINARY_INV)
        ret, finalGray = cv2.threshold(finalGray, 251, 255, cv2.THRESH_BINARY_INV)
        # cv2.imshow("After Threshold", gray)

        # _, contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = contours[1] if imutils.is_cv3() else contours[0]
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
        predictions = []

        for letter_bounding_box in letter_image_regions:
            # Grab the coordinates of the letter in the image
            x, y, w, h = letter_bounding_box

            # Extract the letter from the original image with a 2-pixel margin around the edge
            letter_image = finalGray[y - 2:y + h + 2, x - 2:x + w + 2]

            # Re-size the letter image to 20x20 pixels to match training data
            letter_image = resize_to_fit(letter_image, 20, 20)

            # Turn the single image into a 4d list of images to make Keras happy
            letter_image = np.expand_dims(letter_image, axis=2)
            letter_image = np.expand_dims(letter_image, axis=0)

            # Ask the neural network to make a prediction
            prediction = model.predict(letter_image)

            # Convert the one-hot-encoded prediction back to a normal letter
            letter = lb.inverse_transform(prediction)[0]
            predictions.append(letter)

            # draw the prediction on the output image
            # cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
            # cv2.putText(output, letter, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            # Print the captcha's text
        captcha_text = "".join(predictions)
        codes.append(captcha_text)
        numchars.append(len(captcha_text))
        correct = filename.split(".")[0]
        if (correct != captcha_text):
            # print("CAPTCHA text is: {}".format(captcha_text))
            # print("File name :{}".format(filename))
            more_less_count = more_less_count + 1
        # else:
        #     print("Equal {} {}".format(captcha_text, filename))
        total_count = total_count + 1
        # Show the annotated image
        # cv2.imshow("Output", output)
        # cv2.waitKey()
    return (np.array(numchars) , codes)


def decaptcha( filenames ):
    # numChars = 3 * np.ones( (len( filenames ),) )
    # The use of a model file is just for sake of illustration
    # file = open( "model.txt", "r" )
    # codes = file.read().splitlines()
    # file.close()
    print(filenames)

    (numChars, codes) = solve(filenames)
    return (numChars, codes)