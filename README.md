# Color_Recognition

- Color Correction.ipynb: Color correction by comparing the images captured from usb and csi cameras.
- Test Camera.ipynb: Show captured video in Notebook.
- Naive Implenet.ipynb: Implementation tests.

- scripts
    - hand_utils.py: Detect fingertip using mediapipe. Recognize color in a square patch around the fingertip by majority vote in HSV color space by K-means-clustering.
    - profile_utils.py: Functions for time comsumption evaluation and camera capture using thread.
    - TestHandDetect.py: Run process for a given times to evaluate time consumption.
    - FingerDec.py: Convert detected color to words
