import cv2

max_value = 255
max_value_H = 360//2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'


def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv2.setTrackbarPos(low_H_name, window_detection_name, low_H)


def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv2.setTrackbarPos(high_H_name, window_detection_name, high_H)


def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv2.setTrackbarPos(low_S_name, window_detection_name, low_S)


def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv2.setTrackbarPos(high_S_name, window_detection_name, high_S)


def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv2.setTrackbarPos(low_V_name, window_detection_name, low_V)


def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv2.setTrackbarPos(high_V_name, window_detection_name, high_V)


cv2.namedWindow(window_detection_name)

cv2.createTrackbar(low_H_name, window_detection_name,
    low_H, max_value_H, on_low_H_thresh_trackbar)
cv2.createTrackbar(high_H_name, window_detection_name,
    high_H, max_value_H, on_high_H_thresh_trackbar)
cv2.createTrackbar(low_S_name, window_detection_name,
    low_S, max_value, on_low_S_thresh_trackbar)
cv2.createTrackbar(high_S_name, window_detection_name,
    high_S, max_value, on_high_S_thresh_trackbar)
cv2.createTrackbar(low_V_name, window_detection_name,
    low_V, max_value, on_low_V_thresh_trackbar)
cv2.createTrackbar(high_V_name, window_detection_name,
    high_V, max_value, on_high_V_thresh_trackbar)

img = cv2.imread('videos/test8.png')
cap = cv2.VideoCapture("videos/goml999.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 47000)

# https://docs.opencv.org/3.4.5/da/d97/tutorial_threshold_inRange.html
while True:
    _, frame = cap.read()
    frame = frame[280:, :]
    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # 0.3 - 1.0 ms

    mask = cv2.inRange(frame_HSV, (low_H, low_S, low_V), # 0.1 ms
        (high_H, high_S, high_V))

    res = cv2.bitwise_and(frame, frame, mask=mask) # 0.1 ms
    res_inv = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))

    blur = cv2.GaussianBlur(res, (5, 5), 0) # 0.1 - 0.5 ms
    blur_inv = cv2.GaussianBlur(res_inv, (5, 5), 0)
    #blur = cv2.medianBlur(img, 5)
    #blur = cv2.bilateralFilter(img, 9, 75, 75)


    cv2.imshow('Video Capture', frame)
    cv2.imshow(window_detection_name, mask)
    cv2.imshow('Video Capture AND', res)
    cv2.imshow('Video Capture INV', res_inv)
    cv2.imshow('Video Capture Blur AND', blur)
    cv2.imshow('Video Capture Blur INV', blur_inv)

    key = cv2.waitKey(30)
    if key == ord('q') or key == 27:
        break
