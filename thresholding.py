import cv2

# SmashScan Libraries
import util

# Refer to OpenCV documentation:
# https://docs.opencv.org/3.4.5/da/d97/tutorial_threshold_inRange.html
# https://docs.opencv.org/3.4.5/de/d25/imgproc_color_conversions.html
# https://docs.opencv.org/3.4/da/d6a/tutorial_trackbar.html
# https://docs.opencv.org/3.4.5/da/d97/tutorial_threshold_inRange.html


# An object that creates a parameter analyzer window for HSV ranges, and two
# separate windows that displays the results of the trackbar ranges.
class HsvParamAnalyzer:

    def __init__(self, video_location, start_fnum=0):
        self.window_name = 'Object Detection'
        self.low_H_name = 'Low H'
        self.low_S_name = 'Low S'
        self.low_V_name = 'Low V'
        self.high_H_name = 'High H'
        self.high_S_name = 'High S'
        self.high_V_name = 'High V'

        self.low_H, self.low_S, self.low_V = 0, 0, 0
        self.high_H, self.high_S, self.high_V = 180, 255, 255

        cv2.namedWindow(self.window_name)
        cv2.createTrackbar(self.low_H_name, self.window_name,
            self.low_H, 180, self.on_low_H_thresh_trackbar)
        cv2.createTrackbar(self.high_H_name, self.window_name,
            self.high_H, 180, self.on_high_H_thresh_trackbar)
        cv2.createTrackbar(self.low_S_name, self.window_name,
            self.low_S, 255, self.on_low_S_thresh_trackbar)
        cv2.createTrackbar(self.high_S_name, self.window_name,
            self.high_S, 255, self.on_high_S_thresh_trackbar)
        cv2.createTrackbar(self.low_V_name, self.window_name,
            self.low_V, 255, self.on_low_V_thresh_trackbar)
        cv2.createTrackbar(self.high_V_name, self.window_name,
            self.high_V, 255, self.on_high_V_thresh_trackbar)

        self.capture = cv2.VideoCapture(video_location)
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, start_fnum)


    # The standard test iterates through the entire video with multiple track
    # bars to vary HSV thresholds. Results can be seen in a separate window.
    def standard_test(self):
        while True:
            _, frame = self.capture.read()
            frame = frame[280:, :]
            frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(frame_HSV, (self.low_H, self.low_S, self.low_V),
                (self.high_H, self.high_S, self.high_V))

            res = cv2.bitwise_and(frame, frame, mask=mask)
            res_inv = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))

            cv2.imshow(self.window_name, mask)
            cv2.imshow('Video Capture AND', res)
            cv2.imshow('Video Capture INV', res_inv)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break


    def on_low_H_thresh_trackbar(self, val):
        self.low_H = val
        self.low_H = min(self.high_H-1, self.low_H)
        cv2.setTrackbarPos(self.low_H_name, self.window_name, self.low_H)


    def on_high_H_thresh_trackbar(self, val):
        self.high_H = val
        self.high_H = max(self.high_H, self.low_H+1)
        cv2.setTrackbarPos(self.high_H_name, self.window_name, self.high_H)


    def on_low_S_thresh_trackbar(self, val):
        self.low_S = val
        self.low_S = min(self.high_S-1, self.low_S)
        cv2.setTrackbarPos(self.low_S_name, self.window_name, self.low_S)


    def on_high_S_thresh_trackbar(self, val):
        self.high_S = val
        self.high_S = max(self.high_S, self.low_S+1)
        cv2.setTrackbarPos(self.high_S_name, self.window_name, self.high_S)


    def on_low_V_thresh_trackbar(self, val):
        self.low_V = val
        self.low_V = min(self.high_V-1, self.low_V)
        cv2.setTrackbarPos(self.low_V_name, self.window_name, self.low_V)


    def on_high_V_thresh_trackbar(self, val):
        self.high_V = val
        self.high_V = max(self.high_V, self.low_V+1)
        cv2.setTrackbarPos(self.high_V_name, self.window_name, self.high_V)


# An object that creates a parameter analyzer window for damage OCR.
class DmgParamAnalyzer:

    def __init__(self, video_location, start_fnum=0, stop_fnum=0):
        self.capture = cv2.VideoCapture(video_location)
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, start_fnum)
        self.window_name = "Damage Parameter Analyzer"

        self.start_fnum = start_fnum
        self.stop_fnum = stop_fnum
        if stop_fnum == 0:
            self.stop_fnum = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

        cv2.namedWindow(self.window_name)
        cv2.createTrackbar("Binary/Otsu Thresh", self.window_name,
            0, 1, self.on_thresh_trackbar)

        self.thresh_flag = False


    def standard_test(self):
        for fnum in range(self.start_fnum, self.stop_fnum):
            frame = util.get_frame(self.capture, fnum, gray_flag=True)
            frame = frame[300:340, 80:220]

            if self.thresh_flag:
                _, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
            else:
                _, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_OTSU)

            cv2.imshow(self.window_name, frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    def on_thresh_trackbar(self, val):
        self.thresh_flag = val
