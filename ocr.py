import time
import cv2
import pytesseract
import numpy as np

# SmashScan libraries
import util

# https://github.com/tesseract-ocr/tesseract/wiki/Command-Line-Usage
# 7 - single text line, 8 - single word, 8 works well with background blobs.

def show_ocr_result(frame):
    start_time = time.time()
    text = pytesseract.image_to_string(frame, lang="eng", config="--psm 8")
    print(text)
    util.display_total_time(start_time)

    start_time = time.time()
    pytess_result = pytesseract.image_to_boxes(frame, lang="eng",
        config="--psm 8", output_type=pytesseract.Output.DICT)
    # print(pytess_result)
    util.display_total_time(start_time)

    bbox_list = list()
    for i, _ in enumerate(pytess_result['bottom']):
        tl = (pytess_result['left'][i], pytess_result['bottom'][i])
        br = (pytess_result['right'][i], pytess_result['top'][i])
        bbox_list.append((tl, br))
    util.show_frame(frame, bbox_list=bbox_list, wait_flag=True)

    start_time = time.time()
    pytess_data = pytesseract.image_to_data(frame, lang="eng",
        config="--psm 8", output_type=pytesseract.Output.DICT)
    # print(pytess_data)
    util.display_total_time(start_time)

    bbox_list = list()
    for i, conf in enumerate(pytess_data['conf']):
        if int(conf) != -1:
            print("\tconf: {}".format(conf))
            tl = (pytess_data['left'][i], pytess_data['top'][i])
            br = (tl[0]+pytess_data['width'][i], tl[1]+pytess_data['height'][i])
            bbox_list.append((tl, br))
    util.show_frame(frame, bbox_list=bbox_list, wait_flag=True)


def ocr_test(img, hsv_flag, avg_flag=False, gau_flag=False,
    med_flag=False, bil_flag=False, inv_flag=True):

    # Create a grayscale and HSV copy of the input image.
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # If the HSV flag is enabled, select white OR red -> (High S AND Mid H)'
    if hsv_flag:
        mask = cv2.inRange(img_hsv, (15, 50, 0), (160, 255, 255))
        result_img = cv2.bitwise_and(img_gray, img_gray,
            mask=cv2.bitwise_not(mask))
    else:
        result_img = img_gray

    # Apply a post blurring filter according to the input flag given.
    # https://docs.opencv.org/3.4.5/d4/d13/tutorial_py_filtering.html
    if avg_flag:
        result_img = cv2.blur(result_img, (5, 5))
    elif gau_flag:
        result_img = cv2.GaussianBlur(result_img, (5, 5), 0)
    elif med_flag:
        result_img = cv2.medianBlur(result_img, 5)
    elif bil_flag:
        result_img = cv2.bilateralFilter(result_img, 9, 75, 75)

    # Invert the image to give the image a black on white background.
    if inv_flag:
        result_img = cv2.bitwise_not(result_img)

    display_ocr_test_flags(hsv_flag, avg_flag, gau_flag,
        med_flag, bil_flag, inv_flag)
    show_ocr_result(result_img)


# Display the OCR test flags in a structured format.
def display_ocr_test_flags(hsv_flag, avg_flag, gau_flag,
    med_flag, bil_flag, inv_flag):
    print("hsv_flag={}".format(hsv_flag))

    if avg_flag:
        print("avg_flag={}".format(avg_flag))
    elif gau_flag:
        print("gau_flag={}".format(gau_flag))
    elif med_flag:
        print("med_flag={}".format(med_flag))
    elif bil_flag:
        print("bil_flag={}".format(bil_flag))

    print("inv_flag={}".format(inv_flag))


def contour_test(img):
    _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    img_d = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_d, contours, -1, (255, 0, 0), 2)
    cv2.imshow('test', img_d)
    cv2.waitKey(0)

    for i, contour in enumerate(contours):
        img_d = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_d, contour, -1, (255, 0, 0), 3)

        moment = cv2.moments(contour)
        if moment['m00']: # Removes single points
            cx = int(moment['m10']/moment['m00'])
            cy = int(moment['m01']/moment['m00'])
            print("Center: {}".format((cx, cy)))
            cv2.circle(img_d, (cx, cy), 3, (0, 0, 255), -1)

        print("Area: {}".format(cv2.contourArea(contour)))
        print("Permeter: {} ".format(cv2.arcLength(contour, True)))

        cv2.imshow('test', img_d)
        cv2.waitKey(0)

        mask = np.zeros(img.shape, np.uint8)
        cv2.drawContours(mask, contours, i, 255, cv2.FILLED)
        res = cv2.bitwise_and(img, img, mask=mask)
        cv2.imshow('test', res)
        cv2.waitKey(0)


for fnum in [3400, 5000]:
    capture = cv2.VideoCapture("videos/tbh1.mp4")
    frame = util.get_frame(capture, fnum, gray_flag=True)
    frame = frame[300:340, 80:220]
    cv2.imshow('frame', frame)
    cv2.waitKey(0)

    #frame = cv2.imread('videos/test4.png', cv2.IMREAD_GRAYSCALE)
    #show_ocr_result(frame)

    #img2 = cv2.imread('videos/test4.png', cv2.IMREAD_COLOR)
    #ocr_test(img2, hsv_flag=False)
    #ocr_test(img2, hsv_flag=False, avg_flag=True)
    #ocr_test(img2, hsv_flag=False, gau_flag=True)
    #ocr_test(img2, hsv_flag=False, med_flag=True)
    #ocr_test(img2, hsv_flag=False, bil_flag=True)

    #ocr_test(img2, hsv_flag=True)
    #ocr_test(img2, hsv_flag=True, avg_flag=True)
    #ocr_test(img2, hsv_flag=True, gau_flag=True)
    #ocr_test(img2, hsv_flag=True, med_flag=True)
    #ocr_test(img2, hsv_flag=True, bil_flag=True)

    # https://docs.opencv.org/3.4.5/d7/d4d/tutorial_py_thresholding.html
    print("thresh")
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    th = cv2.medianBlur(thresh, 5)
    show_ocr_result(cv2.bitwise_not(th))
    contour_test(th)

    print("adaothresh")
    _, th2 = cv2.threshold(frame, 0, 255, cv2.THRESH_OTSU)
    show_ocr_result(cv2.bitwise_not(th2))
