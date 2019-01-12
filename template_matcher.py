import cv2
import numpy as np

# SmashScan libraries
import util

# Execute the cv2.matchTemplate algorithm over a video. Parameters include a
# step_size for the speed of video iteration, frame_range for the range of
# frame numbers to be surveyed, gray_flag for a grayscale or BGR analysis,
# roi_flag for only a sub-image (region of interest) to be searched, and
# show_flag which displays results with the cv2.imshow() window.
def tm_test(capture, step_size, frame_range=None,
    gray_flag=True, roi_flag=True, show_flag=False):

    # Set the starting and stopping frame number to search for TM results. If
    # no frame_range parameter was input, iterate through the entire video.
    start_fnum, stop_fnum = 0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_range:
        start_fnum, stop_fnum = frame_range

    # Read the percentage sign image file and extract a binary mask based
    # off of the alpha channel. Also, resize to the 360p base height.
    template_img, template_mask = get_image_and_mask(
        "resources/pct.png", 360/480, gray_flag=gray_flag)
    h, w = template_img.shape[:2]

    # Iterate through video and use cv2 to perform template matching.
    for current_fnum in range(start_fnum, stop_fnum, step_size):
        frame = util.get_frame(capture, current_fnum, gray_flag)

        # Get the confidence and location of cv2.matchTemplate().
        confidence_list, tl_list = get_tm_results(frame, template_img,
            template_mask, roi_flag, num_results=4)

        # Display the frame with an accuracy label if show_flag is enabled.
        if show_flag:
            show_tm_results(frame, confidence_list, tl_list, w, h)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


# Return the confidence and location of the result of cv2.matchTemplate().
# Parameters include num_results to specify the number of templates to search,
# and an roi_flag to only search a sub-region of the frame.
def get_tm_results(frame, template_img, template_mask, roi_flag, num_results):

    # Assuming a Wide 360p format (640×360), only search the bottom quarter
    # of the input frame for the template if the roi_flag is enabled.
    if roi_flag:
        frame = frame[270:, :]

    # Match the template using a normalized cross-correlation method.
    match_mat = cv2.matchTemplate(
        frame, template_img, cv2.TM_CCORR_NORMED, mask=template_mask)

    # Retrieve the confidence and top-left points from the match matrix.
    confidence_list, tl_list = get_match_results(match_mat, num_results)

    # Compensate for point location if a region of interest was used.
    if roi_flag:
        for i in range(num_results):
            tl_list[i] = (tl_list[i][0], tl_list[i][1] + 270)

    return confidence_list, tl_list


# Take the result of cv2.matchTemplate, and find the n-most probable locations
# of a template match. To find multiple locations, the region around a 
# successful match is zeroed. Return a list of the confidences and locations.
def get_match_results(match_mat, num_results):
    max_val_list, top_left_list = list(), list()
    match_mat_dims = match_mat.shape
    for _ in range(0, num_results):
        _, max_val, _, top_left = cv2.minMaxLoc(match_mat)
        set_subregion_to_zeros(match_mat, match_mat_dims, top_left, radius=2)
        max_val_list.append(max_val)
        top_left_list.append(top_left)
    return(max_val_list, top_left_list)


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


# Given an image location, extract the image and alpha (transparent) mask.
def get_image_and_mask(image_location, resize_ratio=None, gray_flag=False):

    # Load image from file with alpha channel (UNCHANGED flag).
    img = cv2.imread(image_location, cv2.IMREAD_UNCHANGED)

    # If an alpha channel does not exist, just return the base image.
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
    if resize_ratio:
        h, w = img.shape[:2]
        h, w = int(h * resize_ratio), int(w * resize_ratio)
        img = cv2.resize(img, (w, h))
        mask = cv2.resize(mask, (w, h))

    return img, mask


# Given a list of confidences and points, call the util.show_frame function
# with the appropriate inputs. Operations include generating bottom-right
# points and joining the multiple confidence values into a single string.
def show_tm_results(frame, confidence_list, top_left_list, width, height):
    bbox_list = list()
    for tl in top_left_list:
        br = (tl[0] + width, tl[1] + height)
        bbox_list.append((tl, br))
    label_list = ["{:0.3f}".format(i) for i in confidence_list]
    label = " ".join(label_list)
    util.show_frame(frame, bbox_list, text=label)
