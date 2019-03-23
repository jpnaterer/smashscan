import time
import cv2
import pytesseract
import numpy as np

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

    def __init__(self, video_location, start_fnum=0, stop_fnum=0):

        self.capture = cv2.VideoCapture(video_location)
        self.start_fnum = start_fnum
        self.stop_fnum = stop_fnum
        if stop_fnum == 0:
            self.stop_fnum = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

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


    # The standard test iterates through the entire video with multiple track
    # bars to vary HSV thresholds. Results can be seen in a separate window.
    def standard_test(self):
        for fnum in range(self.start_fnum, self.stop_fnum):
            frame = util.get_frame(self.capture, fnum)
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


    # A number of methods corresponding to the various trackbars available.
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
        cv2.createTrackbar("Step Size", self.window_name,
            1, 100, self.on_step_trackbar)
        cv2.createTrackbar("Delay", self.window_name,
            10, 500, self.on_delay_trackbar)
        cv2.createTrackbar("Thresh ~ Bin, Otsu", self.window_name,
            0, 1, self.on_thresh_trackbar)
        cv2.createTrackbar("Pre Blur ~ 0, Gaus, Med", self.window_name,
            0, 2, self.on_pre_blur_trackbar)
        cv2.createTrackbar("Post Blur ~ 0, Med", self.window_name,
            0, 1, self.on_post_blur_trackbar)
        cv2.createTrackbar("Contour Filter ~ Off, On", self.window_name,
            0, 1, self.on_contour_trackbar)
        cv2.createTrackbar("Contour Display ~ Off, On", self.window_name,
            0, 1, self.on_contour_disp_trackbar)
        cv2.createTrackbar("Contour Min Area", self.window_name,
            1, 1000, self.on_contour_min_area_trackbar)
        cv2.createTrackbar("Contour Max Area", self.window_name,
            5000, 5000, self.on_contour_max_area_trackbar)
        cv2.createTrackbar("OCR ~ Off, On", self.window_name,
            0, 1, self.on_ocr_trackbar)
        cv2.createTrackbar("OCR ~ 1 Line, 1 Word", self.window_name,
            0, 1, self.on_ocr_mode_trackbar)

        self.step_size = 1
        self.step_delay = 10
        self.pre_blur_val = 0
        self.thresh_flag = False
        self.post_blur_val = 0
        self.contour_flag = False
        self.contour_disp_flag = False
        self.contour_min_area = 1
        self.contour_max_area = 5000
        self.ocr_flag = False
        self.ocr_mode_flag = False


    # The method that must be called to boot up the paramater analysis GUI.
    def standard_test(self):
        fnum = self.start_fnum
        time_queue = list()
        disp_dict = dict()
        while fnum < self.stop_fnum:
            start_time = time.time()
            fnum += self.step_size
            frame = util.get_frame(self.capture, fnum, gray_flag=True)
            frame = frame[300:340, 80:220] # 300:340, 200:320
            frame = self.param_filter(frame)
            if self.contour_flag:
                frame = self.contour_filter(frame)

            if self.ocr_flag:
                conf_text = "--psm 7"       # Single test line mode.
                if self.ocr_mode_flag:      # Single word mode.
                    conf_text = "--psm 8"

                text = pytesseract.image_to_string(cv2.bitwise_not(frame),
                    lang="eng", config=conf_text)
                disp_dict["OCR"] = text

            util.display_pa_fps(start_time, time_queue, disp_dict)
            cv2.imshow(self.window_name, frame)
            if cv2.waitKey(self.step_delay) & 0xFF == ord('q'):
                break


    # Apply filters to frame according to GUI parameters.
    def param_filter(self, frame):
        # Apply pre-blur according to trackbar value.
        if self.pre_blur_val == 1:
            frame = cv2.GaussianBlur(frame, (5, 5), 0)
        elif self.pre_blur_val == 2:
            frame = cv2.medianBlur(frame, 5)

        # Apply a thresholding method according to trackbar value.
        if self.thresh_flag:
            _, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
        else:
            _, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_OTSU)

        # Apply post-blur according to trackbar value.
        if self.post_blur_val:
            frame = cv2.medianBlur(frame, 5)

        return frame


    # Apply filterrs to frame according to contour parameters.
    def contour_filter(self, frame):
        _, contours, _ = cv2.findContours(frame,
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        new_frame = np.zeros(frame.shape, np.uint8)
        for i, contour in enumerate(contours):
            c_area = cv2.contourArea(contour)
            if self.contour_min_area <= c_area <= self.contour_max_area:
                mask = np.zeros(frame.shape, np.uint8)
                cv2.drawContours(mask, contours, i, 255, cv2.FILLED)
                mask = cv2.bitwise_and(frame, mask)
                new_frame = cv2.bitwise_or(new_frame, mask)
        frame = new_frame

        if self.contour_disp_flag:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(frame, contours, -1, (255, 0, 0), 1)

        return frame


    # A number of methods corresponding to the various trackbars available.
    def on_step_trackbar(self, val):
        self.step_size = val

    def on_delay_trackbar(self, val):
        self.step_delay = val

    def on_thresh_trackbar(self, val):
        self.thresh_flag = val

    def on_pre_blur_trackbar(self, val):
        self.pre_blur_val = val

    def on_post_blur_trackbar(self, val):
        self.post_blur_val = val

    def on_contour_trackbar(self, val):
        self.contour_flag = val

    def on_contour_disp_trackbar(self, val):
        self.contour_disp_flag = val

    def on_contour_min_area_trackbar(self, val):
        self.contour_min_area = val

    def on_contour_max_area_trackbar(self, val):
        self.contour_max_area = val

    def on_ocr_trackbar(self, val):
        self.ocr_flag = val

    def on_ocr_mode_trackbar(self, val):
        self.ocr_mode_flag = val
