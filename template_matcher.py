import cv2
import numpy as np

# SmashScan libraries
import util

# An object that takes a capture and a number of input parameters and performs
# a template matching test with the main method tm_test.
class TemplateMatcher:

    def __init__(self, capture, step_size=60, frame_range=None,
        gray_flag=True, roi_flag=True, show_flag=False):

        self.capture = capture
        self.step_size = step_size
        self.frame_range = frame_range
        self.gray_flag = gray_flag
        self.roi_flag = roi_flag
        self.show_flag = show_flag
        self.template_match_radius = 2

        # Read the percentage sign image file and extract a binary mask based
        # off of the alpha channel. Also, resize to the 360p base height.
        self.template_img, self.template_mask = get_image_and_mask(
            "resources/pct.png", 360/480, gray_flag)
        self.template_shape = self.template_img.shape[:2]


    # Execute the cv2.matchTemplate algorithm over a video. Parameters include
    # a step_size for the speed of iteration, frame_range for the range of
    # frame numbers to be surveyed, gray_flag for a grayscale or BGR analysis,
    # roi_flag for only a sub-image (region of interest) to be searched, and
    # show_flag which displays results with the cv2.imshow() window.
    def tm_test(self):

        # Set the start and stop frame number to search for TM results. If no
        # frame_range parameter was input, iterate through the entire video.
        start_fnum = 0
        stop_fnum = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.frame_range:
            start_fnum, stop_fnum = self.frame_range

        # Iterate through video and use cv2 to perform template matching.
        for current_fnum in range(start_fnum, stop_fnum, self.step_size):
            frame = util.get_frame(self.capture, current_fnum, self.gray_flag)

            # Get the confidence and location of cv2.matchTemplate().
            confidence_list, tl_list = self.get_tm_results(frame, 4)

            # Display the frame with an accuracy label if show_flag is enabled.
            if self.show_flag:
                self.show_tm_results(frame, confidence_list, tl_list)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


    # Return the confidence and location of the result of cv2.matchTemplate().
    # Parameters include num_results to specify the number of matches to find.
    def get_tm_results(self, frame, num_results):

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

        return conf_list, tl_list


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


    # Given a list of confidences and points, call the util.show_frame function
    # with the appropriate inputs. Operations include generating bottom-right
    # points and joining the multiple confidence values into a single string.
    def show_tm_results(self, frame, confidence_list, top_left_list):

        # Create a list of bounding boxes (top-left & bottom-right points),
        # using the input template_shape given as (width, height).
        bbox_list = list()
        for tl in top_left_list:
            br = (tl[0] + self.template_shape[1],
                  tl[1] + self.template_shape[0])
            bbox_list.append((tl, br))

        # Create a list of 3 decimal labels, then join by spaces and display.
        label_list = ["{:0.3f}".format(i) for i in confidence_list]
        label = " ".join(label_list)
        util.show_frame(frame, bbox_list, text=label)


#### Functions not inherent by TemplateMatcher Object ##########################

# Given an image location, extract the image and alpha (transparent) mask.
def get_image_and_mask(img_location, img_scale, gray_flag):

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

    # Resize the image and mask based on input value.
    if img_scale:
        h, w = img.shape[:2]
        h, w = int(h * img_scale), int(w * img_scale)
        img = cv2.resize(img, (w, h))
        mask = cv2.resize(mask, (w, h))

    return img, mask


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
