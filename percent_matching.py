import time
import cv2
import numpy as np

# SmashScan Libraries
import position_tools
import timeline
import util

# An object that takes a capture and a number of input parameters and performs
# a number of template matching operations. Parameters include a frame_range
# for the range of frame numbers to be urveyed, gray_flag for a grayscale or
# BGR analysis, show_flag which displays results with cv2.imshow(), and
# wait_flag which waits between frames.
class PercentMatcher:

    def __init__(self, capture, frame_range=None,
        gray_flag=True, save_flag=False, show_flag=False, wait_flag=False):

        self.capture = capture
        self.gray_flag = gray_flag
        self.save_flag = save_flag
        self.show_flag = show_flag

        # Predetermined parameters that have been tested to work best.
        self.calib_w_range = (24, 30) # The possible template width values.
        self.conf_thresh = 0.8        # The cv2 Template Matching conf thresh.
        self.min_match_length_s = 35  # Minimum time of a "match" in seconds.
        self.num_init_frames = 30     # # of frames to init. template size.
        self.num_port_frames = 30     # # of frames to find port each match.
        self.prec_step_size = 2       # Fnum step size during precision sweep.
        self.max_prec_tl_gap_size = 4 # Max size of precise t.l. gaps to fill.
        self.max_tl_gap_size = 4      # Max size of timeline gaps to fill.
        self.roi_y_tolerance = 3      # The size to expand the ROI y-dimensons.
        self.step_size = 60           # Frame number step size during sweep.
        self.template_zero_radius = 2 # Size of match_mat subregion to zero.

        # Paramaters that are redefined later on during initialization.
        self.template_roi = None      # A bounding box to search for templates.

        # Set the start/stop frame to the full video if frame_range undefined.
        if frame_range:
            self.start_fnum, self.stop_fnum = frame_range
        else:
            self.start_fnum = 0
            self.stop_fnum = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        # Set the wait_length for cv2.waitKey. 0 represents waiting, 1 = 1ms.
        if wait_flag:
            self.wait_length = 0
        else:
            self.wait_length = 1

        # Read the percentage sign image file and extract a binary mask based
        # off of the alpha channel. Also, resize to the 360p base height.
        self.orig_pct_img, self.orig_pct_mask = get_image_and_mask(
            "resources/pct.png", gray_flag)
        self.pct_img, self.pct_mask = resize_image_and_mask(
            self.orig_pct_img, self.orig_pct_mask, 360/480)


    #### PERCENT MATCHER TESTS #################################################

    # 1. The PM Sweep Test iterates over the entire video, searching for four
    # default sized percent sprites within each frame.
    def sweep_test(self):

        # Iterate through input video range. During each iteration, fetch the
        # frame and obtain the percent template confidences and bounding boxes.
        start_time = time.time()
        for fnum in range(self.start_fnum, self.stop_fnum, self.step_size):
            frame = util.get_frame(self.capture, fnum, self.gray_flag)
            confidence_list, bbox_list = self.get_tm_results(frame, 4, 0)

            # Display and save frame if the respective flags are enabled.
            if self.show_flag:
                label_list = ["{:0.3f}".format(i) for i in confidence_list]
                label = " ".join(label_list)
                util.show_frame(frame, bbox_list, label,
                    self.save_flag, "output/{:07d}.png".format(fnum))
                if cv2.waitKey(self.wait_length) & 0xFF == ord('q'):
                    break

        # Display the time taken to complete the test.
        frame_count = (self.stop_fnum - self.start_fnum) // self.step_size
        util.display_fps(start_time, frame_count, "Sweep")


    # 2. The PM Calibrate Test iterates over the entire video, comparing a
    # calibrated template size to the default template size. The "calibrated"
    # template size is determined by resizing the template and dermining which
    # resize-operation yields the highest confidence.
    def calibrate_test(self):

        # Iterate through input video range. During each iteration, fetch the
        # frame and obtain the optimal calibrated template size.
        start_time = time.time()
        for fnum in range(self.start_fnum, self.stop_fnum, self.step_size):
            frame = util.get_frame(self.capture, fnum, self.gray_flag)
            bbox, opt_conf, opt_w, opt_h = self.get_calibrate_results(frame)

            # Get the percent sign accuracy according to the default (480, 584)
            # to (360, 640) rescale change from (24, 32) to (18, 24).
            orig_conf_list, _ = self.get_tm_results(frame, 1, 0)

            # Display frame with a confidence label if show_flag is enabled.
            if self.show_flag:
                label = "({}, {}) {:0.3f} -> {:0.3f}".format(
                    opt_w, opt_h, orig_conf_list[0], opt_conf)
                util.show_frame(frame, [bbox], label,
                    self.save_flag, "output/{:07d}.png".format(fnum))
                if cv2.waitKey(self.wait_length) & 0xFF == ord('q'):
                    break

        # Display the time taken to complete the test.
        frame_count = (self.stop_fnum - self.start_fnum) // self.step_size
        util.display_fps(start_time, frame_count, "Calibrate")


    # 3. The PM Initialize Test iterates over a number of random frames to find
    # an expected template size and region of interest. The ROI is a horizontal
    # of the frame based on the y-values that the templates were found.
    def initialize_test(self):

        # Generate random frames to search for a proper template size.
        start_time, opt_w_list, bbox_list = time.time(), list(), list()
        random_fnum_list = np.random.randint(low=self.start_fnum,
            high=self.stop_fnum, size=self.num_init_frames)

        # Iterate through input video range. During each iteration, fetch the
        # frame and obtain the optimal calibrated template size.
        print("(opt_w, opt_h), (bbox), random_fnum, opt_conf")
        for random_fnum in random_fnum_list:
            frame = util.get_frame(self.capture, random_fnum, self.gray_flag)
            bbox, opt_conf, opt_w, opt_h = self.get_calibrate_results(frame)

            # Store the template width if above a confidence threshold.
            if opt_conf > self.conf_thresh:
                opt_w_list.append(opt_w)
                bbox_list.append(bbox)
                print((opt_w, opt_h), bbox, random_fnum, opt_conf)

            # Display frame with a confidence label if show_flag is enabled.
            if self.show_flag:
                orig_conf_list, _ = self.get_tm_results(frame, 1, 0)
                label = "({}, {}) {:0.3f} -> {:0.3f}".format(
                    opt_w, opt_h, orig_conf_list[0], opt_conf)
                util.show_frame(frame, [bbox], label,
                    self.save_flag, "output/{:07d}.png".format(random_fnum))
                if cv2.waitKey(self.wait_length) & 0xFF == ord('q'):
                    break

        # Display the optimal dims, ROI, and time taken to complete the test.
        opt_w, opt_h = self.get_opt_template_dims(opt_w_list)
        self.template_roi = self.get_opt_template_roi(bbox_list)
        print("Optimal Template Size: ({}, {})".format(opt_w, opt_h))
        print("Optimal ROI bbox: {}".format(self.template_roi))
        util.display_fps(start_time, self.num_init_frames, "Initialize")
        if self.show_flag:
            util.show_frame(frame, [self.template_roi], wait_flag=True)


    # 4. The PM timeline Test initializes the template scale, determines a
    # rough estimate of the video timeline (when a percent sign is present),
    # and then obtains a more precise estimate of the video timeline.
    def timeline_test(self):

        # Use a random number of frames to calibrate the percent template size.
        start_time = time.time()
        self.initialize_template_scale()
        util.display_fps(start_time, self.num_init_frames, "Initialize")

        # Iterate through the video to identify when percent is present.
        start_time = time.time()
        pct_timeline = self.get_pct_timeline()
        frame_count = (self.stop_fnum - self.start_fnum) // self.step_size
        util.display_fps(start_time, frame_count, "Initial Sweep")

        # Fill holes in the history timeline list, and filter out timeline
        # sections that are smaller than a particular size.
        clean_timeline = timeline.fill_filter(pct_timeline,
            self.max_tl_gap_size)
        clean_timeline = timeline.size_filter(clean_timeline,
            self.step_size, self.min_match_length_s)
        if self.show_flag:
            timeline.show_plots(pct_timeline, clean_timeline, ["pct found"])

        # Display the frames associated with the calculated match ranges.
        timeline_ranges = timeline.get_ranges(clean_timeline)
        match_ranges = np.multiply(timeline_ranges, self.step_size)
        if self.show_flag:
            util.show_frames(self.capture, match_ranges.flatten())

        # Display the frames associated with the precise match ranges.
        start_time = time.time()
        new_match_ranges = self.get_precise_match_ranges(match_ranges)
        util.display_total_time(start_time, "Cleaning Sweep")
        print("\tMatch Ranges: {:}".format(match_ranges.tolist()))
        print("\tPrecise Match Ranges: {:}".format(new_match_ranges.tolist()))
        if self.show_flag:
            util.show_frames(self.capture, new_match_ranges.flatten())


    #### PERCENT MATCHER SWEEP METHODS #########################################

    # Given a frame, return a confidence list and bounding box list.
    def get_tm_results(self, frame, num_results, conf_thresh=None):

        # Only a specific subregion of the frame is analyzed. If the template
        # ROI has been initialized, take that frame subregion. Otherwise, take
        # the bottom quarter of the frame assuming a W-360p (640x360) format.
        if self.template_roi:
            frame = frame[self.template_roi[0][1]:self.template_roi[1][1], :]
        else:
            frame = frame[270:, :]

        # Set the confidence threshold to the default, if none was input.
        if conf_thresh is None:
            conf_thresh = self.conf_thresh

        # Match the template using a normalized cross-correlation method and
        # retrieve the confidence and top-left points from the result.
        match_mat = cv2.matchTemplate(frame, self.pct_img,
            cv2.TM_CCORR_NORMED, mask=self.pct_mask)
        conf_list, tl_list = self.get_match_results(
            match_mat, num_results, conf_thresh)

        # Compensate for point location for the used region of interest.
        if self.template_roi:
            for i, _ in enumerate(tl_list):
                tl_list[i] = (tl_list[i][0],
                    tl_list[i][1] + self.template_roi[0][1])
        else:
            for i, _ in enumerate(tl_list):
                tl_list[i] = (tl_list[i][0], tl_list[i][1] + 270)

        # Create a list of bounding boxes (top-left & bottom-right points),
        # using the input template_shape given as (width, height).
        bbox_list = list()
        h, w = self.pct_img.shape[:2]
        for tl in tl_list:
            br = (tl[0] + w, tl[1] + h)
            bbox_list.append((tl, br))

        return conf_list, bbox_list


    # Take the result of cv2.matchTemplate, and find the most likely locations
    # of a template match. To find multiple locations, the region around a
    # successful match is zeroed. Return a list of confidences and locations.
    def get_match_results(self, match_mat, num_results, conf_thresh):
        max_val_list, top_left_list = list(), list()
        match_mat_dims = match_mat.shape

        # Find multiple max locations in the input matrix using cv2.minMaxLoc
        # and then zeroing the surrounding region to find the next match.
        for i in range(0, num_results):
            _, max_val, _, top_left = cv2.minMaxLoc(match_mat)
            set_subregion_to_zeros(match_mat, match_mat_dims,
                top_left, radius=self.template_zero_radius)
            max_val_list.append(max_val)
            top_left_list.append(top_left)

        # Remove results that do not meet the confidence threshold.
        conf_list, tl_list = list(), list()
        for i, conf in enumerate(max_val_list):
            if conf > conf_thresh:
                conf_list.append(conf)
                tl_list.append(top_left_list[i])

        return (conf_list, tl_list)


    #### PERCENT MATCHER CALIBRATION METHODS ###################################

    # Resize the original template a number of times to find the dimensions
    # of the template that yield the highest (optimal) confidence. Return the
    # bounding box, confidence value, and optimal template dimensions.
    def get_calibrate_results(self, frame):
        h, w = self.orig_pct_img.shape[:2]
        opt_max_val, opt_top_left, opt_w, opt_h = 0, 0, 0, 0

        # Assuming W-360p (640×360), only search the bottom of the frame.
        frame = frame[270:, :]

        # Iterate over a num. of widths, and rescale the img/mask accordingly.
        for new_w in range(self.calib_w_range[0], self.calib_w_range[1]):
            new_h = int(new_w * h / w)
            pct_img = cv2.resize(self.orig_pct_img, (new_w, new_h))
            pct_mask = cv2.resize(self.orig_pct_mask, (new_w, new_h))

            # Calculate the confidence and location of the current rescale.
            match_mat = cv2.matchTemplate(frame, pct_img,
                cv2.TM_CCORR_NORMED, mask=pct_mask)
            _, max_val, _, top_left = cv2.minMaxLoc(match_mat)

            # Store the results if the confidence is larger than the previous.
            if max_val > opt_max_val:
                opt_max_val, opt_top_left = max_val, top_left
                opt_w, opt_h = new_w, new_h

        # Compensate for point location for the ROI that was used.
        opt_top_left = (opt_top_left[0], opt_top_left[1] + 270)

        # Format the bounding box and return.
        bbox = (opt_top_left, (opt_top_left[0]+opt_w, opt_top_left[1]+opt_h))
        return bbox, opt_max_val, opt_w, opt_h


    # Given a list of expected widths, return the optimal dimensions of the
    # template bounding box by calculating the median of the list.
    def get_opt_template_dims(self, opt_w_list):
        opt_w = int(np.median(opt_w_list))
        h, w = self.orig_pct_img.shape[:2]
        return (opt_w, round(h*opt_w/w))


    # Given a list of expected bounding boxes, return the optimal region of
    # interest bounding box, that covers a horizontal line over the entire 360p
    # frame. The bounding box must not surpass the boundaries of the frame.
    def get_opt_template_roi(self, bbox_list):
        y_min_list, y_max_list = list(), list()
        for bbox in bbox_list:
            y_min_list.append(bbox[0][1])
            y_max_list.append(bbox[1][1])
        y_min = max(0, min(y_min_list) - self.roi_y_tolerance)
        y_max = min(359, max(y_max_list) + self.roi_y_tolerance)
        return ((0, y_min), (639, y_max))


    #### PERCENT MATCHER INITIALIZE METHODS ####################################

    # Selects a random number of frames to calibrate the percent template size.
    def initialize_template_scale(self):

        # Generate random frames to search for a proper template size.
        random_fnum_list = np.random.randint(low=self.start_fnum,
            high=self.stop_fnum, size=self.num_init_frames)
        opt_w_list, bbox_list = list(), list()

        # Iterate through input video range. During each iteration, fetch the
        # frame and obtain the optimal calibrated template size.
        for random_fnum in random_fnum_list:
            frame = util.get_frame(self.capture, random_fnum, self.gray_flag)
            bbox, opt_conf, opt_w, _ = self.get_calibrate_results(frame)

            # Store template info if confidence above an input threshold.
            if opt_conf > self.conf_thresh:
                opt_w_list.append(opt_w)
                bbox_list.append(bbox)

        # Calculate the median of the optimal widths and rescale accordingly.
        opt_w, opt_h = self.get_opt_template_dims(opt_w_list)
        self.pct_img = cv2.resize(self.orig_pct_img, (opt_w, opt_h))
        self.pct_mask = cv2.resize(self.orig_pct_mask, (opt_w, opt_h))

        # Calculate the region of interest to search for the template.
        self.template_roi = self.get_opt_template_roi(bbox_list)


    #### PERCENT MATCHER TIMELINE METHODS ######################################

    # Iterate through the video to identify when the percent sprite is present.
    def get_pct_timeline(self):

        pct_timeline = list()
        for fnum in range(self.start_fnum, self.stop_fnum, self.step_size):
            # Obtain the frame and get the template confidences and locations.
            frame = util.get_frame(self.capture, fnum, self.gray_flag)
            confidence_list, _ = self.get_tm_results(frame, 1)

            # Append to the percent timeline according to if percent was found.
            if confidence_list:
                pct_timeline.append(0)
            else:
                pct_timeline.append(-1)

        return pct_timeline


    # Given an initial guess of match ranges, make a more precise estimate.
    def get_precise_match_ranges(self, init_match_ranges):

        # Iterate through the match ranges, going backwards if at the start
        # of a match, and going forward if at the end of a match.
        prec_match_ranges_flat = list()
        init_match_ranges_flat = init_match_ranges.flatten()
        for i, fnum_prediction in enumerate(init_match_ranges_flat):
            fnum = fnum_prediction
            if i % 2 == 0:
                current_step_size = -self.prec_step_size
            else:
                current_step_size = self.prec_step_size

            # Iterate through the video using fnum until no percent has been
            # found for a specified number of frames.
            while True:
                frame = util.get_frame(self.capture, fnum, self.gray_flag)
                confidence_list, _ = self.get_tm_results(frame, 1)

                # Increment the precise counter if no pct was found.
                if confidence_list:
                    prec_counter = 0
                else:
                    prec_counter += 1

                # Exit if there has been no percent found over multiple frames.
                if prec_counter == self.max_prec_tl_gap_size:
                    prec_match_ranges_flat.append(
                        fnum - current_step_size*(prec_counter+1))
                    break
                elif fnum == 0 or fnum >= self.stop_fnum - self.prec_step_size:
                    prec_match_ranges_flat.append(fnum)
                    break
                fnum = fnum + current_step_size

        # Return the match ranges as a list of pairs.
        return np.reshape(prec_match_ranges_flat, (-1, 2))


    #### PERCENT MATCHER EXTERNAL METHODS ######################################

    # Run the timeline template test over a video range.
    def get_match_ranges(self):

        # Use a random number of frames to calibrate the percent template size.
        self.initialize_template_scale()

        # Iterate through the video to identify when percent is present.
        pct_timeline = self.get_pct_timeline()

        # Fill holes in the history timeline list, and filter out timeline
        # sections that are smaller than a particular size.
        clean_timeline = timeline.fill_filter(pct_timeline,
            self.max_tl_gap_size)
        clean_timeline = timeline.size_filter(clean_timeline,
            self.step_size, self.min_match_length_s)

        # Display the frames associated with the calculated match ranges.
        timeline_ranges = timeline.get_ranges(clean_timeline)
        match_ranges = np.multiply(timeline_ranges, self.step_size)

        # Display the frames associated with the precise match ranges.
        new_match_ranges = self.get_precise_match_ranges(match_ranges)
        return new_match_ranges.tolist()


    # Given a list of match ranges and bboxes, return the ports in use.
    def get_port_num_list(self, match_ranges, match_bboxes):

        start_time = time.time()
        for i, match_range in enumerate(match_ranges):
            random_fnum_list = np.random.randint(low=match_range[0],
                high=match_range[1], size=self.num_port_frames)

            # print(match_range)
            x_pos_list = list()
            for fnum in random_fnum_list:
                frame = util.get_frame(self.capture, fnum, self.gray_flag)
                _, bbox_list = self.get_tm_results(frame, 4)
                for bbox in bbox_list:
                    x_pos_list.append(bbox[0][0])
                # print((fnum, conf_list, bbox_list))

            port_pos_list = position_tools.get_port_pos_list(x_pos_list)
            port_num_list = position_tools.get_port_num_list(
                port_pos_list, match_bboxes[i])
            # print(port_num_list)

        util.display_total_time(start_time, "Port Sweep")
        return port_num_list


#### Functions not inherent by PercentMatcher Object ##########################

# Given an image location, extract the image and alpha (transparent) mask.
def get_image_and_mask(img_location, gray_flag):

    # Load image from file with alpha channel (UNCHANGED flag). If an alpha
    # channel does not exist, just return the base image.
    img = cv2.imread(img_location, cv2.IMREAD_UNCHANGED)
    if img.shape[2] <= 3:
        return img, None

    # Create an alpha channel matrix  with values between 0-255. Then
    # threshold the alpha channel to create a binary mask.
    channels = cv2.split(img)
    mask = np.array(channels[3])
    _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)

    # Convert image and mask to grayscale or BGR based on input flag.
    if gray_flag:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    return img, mask


# Resize an image and mask based on an input scale ratio.
def resize_image_and_mask(img, mask, img_scale):
    h, w = img.shape[:2]
    h, w = int(h * img_scale), int(w * img_scale)
    resized_img = cv2.resize(img, (w, h))
    resized_mask = cv2.resize(mask, (w, h))
    return resized_img, resized_mask


# Take a matrix and coordinate, and zero the region around that coordinate.
# This also prevents matrix out of bound errors if the input coordinate is
# near the border. Also, the input coordinate is organized as (x, y). Matrices
# are pass by reference, so the input can be directly modified.
def set_subregion_to_zeros(input_mat, mat_dims, center_pt, radius):

    # Set the top-left and bot-right points of the zeroed region. Note that
    # mat_dims is organized as (height, width), and tl/br is (y, x).
    tl = (max(center_pt[1]-radius, 0),
          max(center_pt[0]-radius, 0))
    br = (min(center_pt[1]+radius+1, mat_dims[0]),
          min(center_pt[0]+radius+1, mat_dims[1]))

    # Calculate the size of the region to be zeroed.
    x_size = br[0] - tl[0]
    y_size = br[1] - tl[1]

    input_mat[tl[0]:br[0], tl[1]:br[1]] = np.zeros((x_size, y_size))
