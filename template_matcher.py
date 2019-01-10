import cv2
import numpy as np

# SmashScan libraries
import util

# Execute the cv2.matchTemplate algorithm over a video. Parameters include a
# step_size for the speed of video iteration, frame_range for the range of
# frame numbers to be surveyed, gray_flag for a grayscale or BGR analysis,
# roi_flag for only a sub-image (region of interest) to be searched, and
# show_flag which displays results with the cv2.imshow() window.
def show_tm_results(capture, step_size, frame_range=None,
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
        confidence, tl = get_tm_result(frame, template_img, 
            template_mask, roi_flag=roi_flag)

        # Display the frame with an accuracy label if show_flag is enabled.
        if show_flag:
            br = (tl[0] + w, tl[1] + h)
            label = "{:0.4f}".format(confidence)
            util.show_frame(frame, bbox=[tl, br], text=label)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


# Return the confidence and location of the result of cv2.matchTemplate().
# Parameters include an roi_flag to only search a sub-region of the frame.
def get_tm_result(frame, template_img, template_mask, roi_flag=True):

    # Assuming a Wide 360p format (640×360), only search the bottom quarter
    # of the input frame for the template if the roi_flag is enabled.
    if roi_flag:
        frame = frame[270:, :]

    # Match the template using a normalized cross-correlation method.
    match_mat = cv2.matchTemplate(
        frame, template_img, cv2.TM_CCORR_NORMED, mask=template_mask)
    _, max_val, _, top_left = cv2.minMaxLoc(match_mat)

    # Compensate for point location if a region of interest was used.
    if roi_flag:
        top_left = (top_left[0], top_left[1] + 270)

    return max_val, top_left


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
