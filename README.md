# Solving_Captchas
To run these scripts, you need the following installed:

-- Python 3
-- Keras : 2.3.0
-- Tensorflow : 1.14
-- opencv
-- pickle
-- imutils


### Step 1: Extract single letters from CAPTCHA images

Run:

python3 extract_single_letters_from_captchas.py

The results will be stored in the "extracted_letter_images" folder.


### Step 2: Train the neural network to recognize single letters

Run:

python3 train_model.py

This will write out "captcha_model.hdf5" and "model_labels.dat"

Step3:

Run: 

python3 eval.py

-- test folder has test dataset just to give an idea on what type of data the model was trained on.