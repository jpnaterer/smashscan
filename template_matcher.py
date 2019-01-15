import cv2
import statistics
import numpy as np

# SmashScan libraries
import util

# An object that takes a capture and a number of input parameters and performs
# a number of template matching tests. Parameters include a step_size for the
# speed of iteration, frame_range for the range of frame numbers to be
# surveyed, gray_flag for a grayscale or BGR analysis, roi_flag for only a
# sub-image (region of interest) to be searched, show_flag which displays
# results with cv2.imshow(), and wait_flag which waits between frames.
class TemplateMatcher:

    def __init__(self, capture, step_size=60, frame_range=None, num_frames=60,
        gray_flag=True, roi_flag=True, show_flag=False, wait_flag=False):

        self.capture = capture
        self.step_size = step_size
        self.num_frames = num_frames
        self.gray_flag = gray_flag
        self.roi_flag = roi_flag
        self.show_flag = show_flag

        # Predetermined parameters that have been tested to work best.
        self.template_match_radius = 2
        self.conf_threshold = 0.8
        self.roi_y_tolerance = 3

        # Set the wait_length for cv2.waitKey. 0 represents waiting, 1 = 1ms.
        if wait_flag:
            self.wait_length = 0
        else:
            self.wait_length = 1

        # Set the start and stop frame number to search for TM results. If no
        # stop_fnum parameter was input, iterate through the entire video.
        self.start_fnum = 0
        self.stop_fnum = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_range[1]:
            self.start_fnum, self.stop_fnum = frame_range

        # Read the percentage sign image file and extract a binary mask based
        # off of the alpha channel. Also, resize to the 360p base height.
        self.orig_template_img, self.orig_template_mask = get_image_and_mask(
            "resources/pct.png", gray_flag)
        self.template_img, self.template_mask = resize_image_and_mask(
            self.orig_template_img, self.orig_template_mask, 360/480)


    #### TEMPLATE MATCHER RUNTIME METHODS ######################################

    def initialize_template_scale(self):
         # Generate random frames to search for a proper template size.
        random_fnum_list = np.random.randint(low=self.start_fnum,
            high=self.stop_fnum, size=self.num_frames)
        max_w_list, bbox_list = list(), list()

        for random_fnum in random_fnum_list:

            # Get the calibrated accuracy for the random frame.
            frame = util.get_frame(self.capture, random_fnum, self.gray_flag)
            bbox, opt_conf, max_w, _ = self.get_tmc_results(frame)

            # Store template info if confidence above an input threshold.
            if opt_conf > self.conf_threshold:
                max_w_list.append(max_w)
                bbox_list.append(bbox)

        # Calculate the Median of the optimal widths and rescale accordingly.
        opt_w = int(np.median(max_w_list))
        h, w = self.orig_template_img.shape[:2]
        opt_h = h*opt_w//w
        self.template_img = cv2.resize(self.orig_template_img, (opt_w, opt_h))
        self.template_mask = cv2.resize(self.orig_template_mask, (opt_w, opt_h))

        # Calculate the region of interest to search for the template.
        print(self.get_template_roi(bbox_list))


    #### TEMPLATE MATCHER TESTS ################################################

    # Run the standard template matching test over a video range.
    def standard_test(self):

        # Iterate through video range and use cv2 to perform template matching.
        for current_fnum in range(self.start_fnum,
            self.stop_fnum, self.step_size):

            # Obtain the frame and get the template confidences and locations.
            frame = util.get_frame(self.capture, current_fnum, self.gray_flag)
            confidence_list, bbox_list = self.get_tms_results(frame, 4)

            # Display frame with a confidence label if show_flag is enabled.
            if self.show_flag:
                label_list = ["{:0.3f}".format(i) for i in confidence_list]
                label = " ".join(label_list)
                util.show_frame(frame, bbox_list, text=label)
                if cv2.waitKey(self.wait_length) & 0xFF == ord('q'):
                    break


    # Run the calibrate template test over a video range.
    def calibrate_test(self):

        # Iterate through video range and use cv2 to perform template matching.
        for current_fnum in range(self.start_fnum,
            self.stop_fnum, self.step_size):

            # Obtain the frame and get the calibrated template size.
            frame = util.get_frame(self.capture, current_fnum, self.gray_flag)
            bbox, opt_conf, max_w, max_h = self.get_tmc_results(frame)

            # Get the percent sign accuracy according to the default (480, 584)
            # to (360, 640) rescale change from (24, 32) to (18, 24).
            orig_conf_list, _ = self.get_tms_results(frame, 1)

            # Display frame with a confidence label if show_flag is enabled.
            if self.show_flag:
                label = "({}, {}) {:0.3f} -> {:0.3f}".format(
                    max_w, max_h, orig_conf_list[0], opt_conf)
                util.show_frame(frame, bbox_list=[bbox], text=label)
                if cv2.waitKey(self.wait_length) & 0xFF == ord('q'):
                    break


    # Run the initialize template test over a number of random frames.
    def initialize_test(self):

        # Generate random frames to search for a proper template size.
        random_fnum_list = np.random.randint(low=self.start_fnum,
            high=self.stop_fnum, size=self.num_frames)
        max_w_list, bbox_list = list(), list()

        for random_fnum in random_fnum_list:

            # Get the calibrated accuracy, and get the original accuracy
            # according to the default (24, 32) to (18, 24) rescale.
            frame = util.get_frame(self.capture, random_fnum, self.gray_flag)
            bbox, opt_conf, max_w, max_h = self.get_tmc_results(frame)
            orig_conf_list, _ = self.get_tms_results(frame, 1)

            # Store the template width if above a confidence threshold.
            if opt_conf > self.conf_threshold:
                max_w_list.append(max_w)
                bbox_list.append(bbox)
                print((max_w, max_h), bbox, random_fnum, opt_conf)

            # Display frame with a confidence label if show_flag is enabled.
            if self.show_flag:
                label = "({}, {}) {:0.3f} -> {:0.3f}".format(
                    max_w, max_h, orig_conf_list[0], opt_conf)
                util.show_frame(frame, bbox_list=[bbox], text=label)
                if cv2.waitKey(self.wait_length) & 0xFF == ord('q'):
                    break

        opt_w = int(np.median(max_w_list))
        h, w = self.template_img.shape[:2]
        print("Optimal Template Size: ({}, {})".format(opt_w, h*opt_w//w))
        print("Optimal ROI bbox: {}".format(self.get_template_roi(bbox_list)))


    #### TEMPLATE MATCHER HELPER METHODS #######################################

    # Return the confidence list and point list required by the TMS test.
    def get_tms_results(self, frame, num_results):

        # Assuming a Wide 360p format (640×360), only search the bottom quarter
        # of the input frame for the template if the roi_flag is enabled.
        if self.roi_flag:
            frame = frame[270:, :]

        # Match the template using a normalized cross-correlation method.
        match_mat = cv2.matchTemplate(frame, self.template_img,
            cv2.TM_CCORR_NORMED, mask=self.template_mask)

        # Retrieve the confidence and top-left points from the match matrix.
        conf_list, tl_list = self.get_match_results(match_mat, num_results)

        # Compensate for point location if a region of interest was used.
        if self.roi_flag:
            for i in range(num_results):
                tl_list[i] = (tl_list[i][0], tl_list[i][1] + 270)

        # Create a list of bounding boxes (top-left & bottom-right points),
        # using the input template_shape given as (width, height).
        bbox_list = list()
        h, w = self.template_img.shape[:2]
        for tl in tl_list:
            br = (tl[0] + w, tl[1] + h)
            bbox_list.append((tl, br))

        return conf_list, bbox_list


    # Take the result of cv2.matchTemplate, and find the most likely locations
    # of a template match. To find multiple locations, the region around a
    # successful match is zeroed. Return a list of confidences and locations.
    def get_match_results(self, match_mat, num_results):
        max_val_list, top_left_list = list(), list()
        match_mat_dims = match_mat.shape

        # Find multiple max locations in the input matrix using cv2.minMaxLoc
        # and then zeroing the surrounding region to find the next match.
        for _ in range(0, num_results):
            _, max_val, _, top_left = cv2.minMaxLoc(match_mat)
            set_subregion_to_zeros(match_mat, match_mat_dims,
                top_left, radius=self.template_match_radius)
            max_val_list.append(max_val)
            top_left_list.append(top_left)

        return(max_val_list, top_left_list)


    # Return the bounding box and label required by the TMC test.
    def get_tmc_results(self, frame):
        h, w = self.orig_template_img.shape[:2]

        # Assuming a Wide 360p format (640×360), only search the bottom quarter
        # of the input frame for the template if the roi_flag is enabled.
        if self.roi_flag:
            frame = frame[270:, :]

        # Get the percent sign accuracy according to the optimal rescale.
        opt_max_val, opt_top_left, max_w, max_h = \
            self.get_calibrate_results(frame, w, h)

        # Compensate for point location if a region of interest was used.
        if self.roi_flag:
            opt_top_left = (opt_top_left[0], opt_top_left[1] + 270)

        # Format the bounding box and return.
        bbox = (opt_top_left, (opt_top_left[0]+max_w, opt_top_left[1]+max_h))
        return bbox, opt_max_val, max_w, max_h


    # Resize the original template a number of times to find the dimensions
    # of the template that yield the highest (optimal) confidence.
    def get_calibrate_results(self, frame, w, h):
        opt_max_val, opt_top_left, opt_w, opt_h = 0, 0, 0, 0

        # Iterate over a num. of widths, and rescale the img/mask accordingly.
        for new_w in range(w - 10, w + 1):
            new_h = int(new_w * h / w)
            template_img = cv2.resize(self.orig_template_img, (new_w, new_h))
            template_mask = cv2.resize(self.orig_template_mask, (new_w, new_h))

            # Calculate the confidence and location of the current rescale.
            match_mat = cv2.matchTemplate(frame, template_img,
                cv2.TM_CCORR_NORMED, mask=template_mask)
            _, max_val, _, top_left = cv2.minMaxLoc(match_mat)

            # Store the results if the confidence is larger than the previous.
            if max_val > opt_max_val:
                opt_max_val, opt_top_left = max_val, top_left
                opt_w, opt_h = new_w, new_h

        return opt_max_val, opt_top_left, opt_w, opt_h


    # Given a list of expected bounding boxes, return a region of interest
    # bounding box, that covers a horizontal line over the entire 360p frame.
    # The bottom y-coordinate must not surpass the boundaries of the frame.
    def get_template_roi(self, bbox_list):
        tol, y_min_list, y_max_list = self.roi_y_tolerance, list(), list()
        for bbox in bbox_list:
            y_min_list.append(bbox[0][1])
            y_max_list.append(bbox[1][1])
        y_min = max(0, min(y_min_list)-tol)
        y_max = min(639, max(y_max_list)+tol)
        return ((0, y_min), (359, y_max))


#### Functions not inherent by TemplateMatcher Object ##########################

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
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

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


# Take a matrix and coordinate, and set the region around that coordinate
# to zeros. This function also prevents matrix out of bound errors if the
# input coordinate is near the matrix border. Also, the input coordinate
# is organized as (row, column) while matrices are organized (x, y). Matrices
# are pass by reference, so the input can be directly modified.
def set_subregion_to_zeros(input_mat, mat_dims, center_pt, radius):

    # Set the top-left and bot-right points of the zeroed region. Note that
    # mat_dims is organized as (width, height) or (x-range,y-range).
    tl = (max(center_pt[1]-radius, 0),
          max(center_pt[0]-radius, 0))
    br = (min(center_pt[1]+radius+1, mat_dims[0]-1),
          min(center_pt[0]+radius+1, mat_dims[1]-1))

    # Calculate the size of the region to be zeroed. Initialize it as a square
    # of size (2r+1), then subtract off the region that is cutoff by a border.
    x_size, y_size = radius*2+1, radius*2+1
    if center_pt[1]-radius < 0:
        x_size -= radius-center_pt[1]
    elif center_pt[1]+radius+1 > mat_dims[0]-1:
        x_size -= center_pt[1]+radius+1 - (mat_dims[0]-1)
    if center_pt[0]-radius < 0:
        y_size -= radius-center_pt[0]
    elif center_pt[0]+radius+1 > mat_dims[1]-1:
        y_size -= center_pt[0]+radius+1 - (mat_dims[1]-1)

    input_mat[tl[0]:br[0], tl[1]:br[1]] = np.zeros((x_size, y_size))
