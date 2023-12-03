# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 15:53:11 2019

@author: Haseeb
"""

from __future__ import print_function
import cv2
import numpy as np
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import os
import shutil
from tensorflow.keras.models import load_model
from PythonScripts import  meter_no_recognizer
from PythonScripts import globals
import pytesseract

def check_alignment(img):
    """
    Check if the given image is aligned using template matching.

    Args:
        img (numpy.ndarray): Input image.

    Returns:
        int: 1 if aligned, 0 otherwise.
    """
    threshold = 0.50
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('PreprocessingTemplate/template_matching.jpg', 0)
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    check_aligned = int(np.any(res >= threshold))
    return check_aligned


def decode(l):
    l=l.tolist()
    return l.index(max(l))

def recognition(img):
    """
    Perform recognition on the input image and append the result to the global list.

    Args:
        img (numpy.ndarray): Input image.
    """
    IMG_SIZE = 50
    new_array = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    result = decode(globals.cnn_model.predict(img)[0])
    globals.list.append(result)




def align_images(im1, im2):
    """
    Align two images using ORB features and homography transformation.

    Args:
        im1 (numpy.ndarray): First input image.
        im2 (numpy.ndarray): Second input image.

    Returns:
        tuple: A tuple containing the aligned image and the homography matrix.
    """
    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(globals.MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    num_good_matches = int(len(matches) * globals.GOOD_MATCH_PERCENT)
    matches = matches[:num_good_matches]

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1_reg = cv2.warpPerspective(im1, h, (width, height))

    return im1_reg, h

def align_images(im1, im2):
    """
    Align two images using ORB features and homography transformation.

    Args:
        im1 (numpy.ndarray): First input image.
        im2 (numpy.ndarray): Second input image.

    Returns:
        tuple: A tuple containing the aligned image and the homography matrix.
    """
    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(globals.MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    num_good_matches = int(len(matches) * globals.GOOD_MATCH_PERCENT)
    matches = matches[:num_good_matches]

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1_reg = cv2.warpPerspective(im1, h, (width, height))

    return im1_reg, h


def most_frequent(L):
    """
    Find the most frequent element and its count in the given list.

    Args:
        L (list): Input list.

    Returns:
        tuple: A tuple containing the most frequent element and its count.
    """
    if not L:
        return 0, 0

    counter = 0
    num = L[0]

    for i in L:
        curr_frequency = L.count(i)
        if curr_frequency >= counter:
            counter = curr_frequency
            num = i

    return num, counter


def process_video(file):
    """
    Process a video file, perform frame processing, and return results.

    Args:
        file (str): Path to the video file.

    Returns:
        tuple: A tuple containing the processed image, meter number, and meter reading.
    """
    cap = cv2.VideoCapture(file)

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        globals.total_frames += 1

        scale_percent = 95

        if (globals.total_frames % 2) == 0:
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            frame_process(frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
        globals.string = ' '
        globals.list = []
        globals.type = -1
        mf, count = most_frequent(globals.results)
        if (count > 3):
            meter_no=most_frequent(globals.results_meterno)[0]
            cnt, image = globals.dic[mf]
            cv2.drawContours(image, [cnt], -1, (0, 255, 0), 3)
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00']) - 57
            cv2.putText(image, text=mf, org=(cx, cy),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0),
                        thickness=2, lineType=cv2.LINE_AA)
            cap.release()
            cv2.destroyAllWindows()
            globals.list = []
            globals.results = []
            globals.results_meterno = []
            globals.dic = {}
            globals.type = -1
            globals.string = ''
            return image , meter_no , mf
            break



def reset_globals():
    """
    Reset global variables in the globals module.
    """
    globals.string = ''
    globals.list = []
    globals.type = -1
    globals.results = []
    globals.results_meterno = []
    globals.dic = {}

def frame_process(frame):
    """
    Process a single frame, extract information, and update global variables.

    Args:
        frame (numpy.ndarray): Input frame to be processed.
    """
    ref_filename = "PreprocessingTemplate/alignment_sample.jpg"
    im_reference = cv2.imread(ref_filename, cv2.IMREAD_COLOR)

    # Read the input image
    im = frame

    # Perform image alignment
    im_reg, h = align_images(im, im_reference)

    # Check alignment result
    d = check_alignment(im_reg)
    if d == 1:
        # Update global variable
        globals.frames_processed += 1

        # Extract meter number region
        row, column, _ = im_reg.shape
        a = int(row * 0.63)
        b = int(row * 0.72)
        c = int(column * 0.57)
        d = int(column * 0.80)
        meter_no = im_reg[a:b, c:d]

        # Process meter number using Tesseract
        result_meterno = meter_no_recognizer.meterno_processing(meter_no)
        if len(result_meterno) != 0:
            globals.results_meterno.append(result_meterno)

        # Resize and process the image
        image = imutils.resize(im_reg, height=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 200, 255)

        # Find contours in the edge map
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        display_cnt = None

        # Loop over the contours to find the thermostat display
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                display_cnt = approx
                break

        # Extract and transform the thermostat display
        warped = four_point_transform(gray, display_cnt.reshape(4, 2))
        l, w = warped.shape
        output = four_point_transform(image, display_cnt.reshape(4, 2))

        # Resize and threshold the transformed image
        scale_percent = 300
        width = int(warped.shape[1] * scale_percent / 100)
        height = int(warped.shape[0] * scale_percent / 100)
        dim = (width, height)
        warped = cv2.resize(warped, dim, interpolation=cv2.INTER_AREA)
        l, w = warped.shape

        dim = (1552, 454)
        warped = cv2.resize(warped, dim, interpolation=cv2.INTER_AREA)
        blur = cv2.GaussianBlur(warped, (5, 5), 0)
        ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Process segments of the image
        rows, col = warped.shape
        x1 = int(rows * 0.10)
        x2 = int(rows * 0.95)
        a = 71
        b = 85
        i = 0
        while a > 0:
            y1 = int(col * (a / 100))
            y2 = int(col * (b / 100))
            crop = warped[x1:x2, y1:y2]

            # Find Canny edges and perform recognition
            blur = cv2.GaussianBlur(crop, (5, 5), 0)
            ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            th3 = cv2.dilate(th3, kernel, iterations=1)
            recognition(th3)
            cv2.waitKey()
            b = a + 2
            a = b - 12
            i += 1

        # Reverse and concatenate the processed results
        globals.list.reverse()
        for i in globals.list:
            if i != 10:
                globals.string += str(i)
        temp = len(globals.string)

        # Update global variables
        globals.results.append(globals.string)
        globals.dic[globals.string] = [display_cnt, image]
        globals.string = ""






