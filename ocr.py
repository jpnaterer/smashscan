import time
import cv2
import pytesseract

# SmashScan libraries
import util

# https://github.com/tesseract-ocr/tesseract/wiki/Command-Line-Usage
# 7 - single text line, 8 - single word

def show_ocr_result(frame):
    start_time = time.time()
    text = pytesseract.image_to_string(frame, lang="eng", config="--psm 7")
    print(text)
    util.display_total_time(start_time)

    start_time = time.time()
    pytess_result = pytesseract.image_to_boxes(frame, lang="eng",
        config="--psm 7", output_type=pytesseract.Output.DICT)
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
        config="--psm 7", output_type=pytesseract.Output.DICT)
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


def ocr_test(img, hsv_flag, inv_flag, gau_flag):
    # cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if hsv_flag:
        mask = cv2.inRange(img_hsv, (40, 60, 0), (160, 255, 255))
        result_img = cv2.bitwise_and(img_gray, img_gray,
            mask=cv2.bitwise_not(mask))
    else:
        result_img = img_gray

    if inv_flag:
        result_img = cv2.bitwise_not(result_img)
    if gau_flag:
        result_img = cv2.GaussianBlur(result_img, (5, 5), 0)

    print("hsv_flag={}".format(inv_flag))
    print("inv_flag={}".format(inv_flag))
    print("gau_flag={}".format(gau_flag))
    show_ocr_result(result_img)


capture = cv2.VideoCapture("videos/g6_1.mp4")
frame = util.get_frame(capture, 1000)

img = cv2.imread('videos/test6.png', cv2.IMREAD_GRAYSCALE)
show_ocr_result(img)

img2 = cv2.imread('videos/test6.png', cv2.IMREAD_COLOR)
ocr_test(img2, hsv_flag=True, inv_flag=False, gau_flag=False)
ocr_test(img2, hsv_flag=True, inv_flag=True, gau_flag=False)
ocr_test(img2, hsv_flag=True, inv_flag=False, gau_flag=True)
ocr_test(img2, hsv_flag=True, inv_flag=True, gau_flag=True)
ocr_test(img2, hsv_flag=False, inv_flag=False, gau_flag=False)
ocr_test(img2, hsv_flag=False, inv_flag=True, gau_flag=False)
ocr_test(img2, hsv_flag=False, inv_flag=False, gau_flag=True)
ocr_test(img2, hsv_flag=False, inv_flag=True, gau_flag=True)

# https://docs.opencv.org/3.4.5/d7/d4d/tutorial_py_thresholding.html
print("thresh")
blur = cv2.GaussianBlur(img, (5, 5), 0)
_, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
thresh = cv2.medianBlur(thresh, 5)
show_ocr_result(cv2.bitwise_not(thresh))

print("adaothresh")
_, th4 = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
show_ocr_result(cv2.bitwise_not(th4))

print("bluradaothresh")
blur = cv2.GaussianBlur(img, (5, 5), 0)
_, th5 = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)
show_ocr_result(cv2.bitwise_not(th5))

# https://docs.opencv.org/3.4.5/d4/d13/tutorial_py_filtering.html
print("blur")
blur = cv2.blur(img, (5, 5))
show_ocr_result(cv2.bitwise_not(blur))

print("blurg")
blur = cv2.GaussianBlur(img, (5, 5), 0)
show_ocr_result(cv2.bitwise_not(blur))

print("med")
median = cv2.medianBlur(img, 5)
show_ocr_result(cv2.bitwise_not(blur))

print("bil")
blur = cv2.bilateralFilter(img, 9, 75, 75)
show_ocr_result(cv2.bitwise_not(blur))
