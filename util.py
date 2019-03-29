import time
import cv2
import numpy as np

# Given a frame number and additional parameters, return a frame.
def get_frame(capture, frame_num, gray_flag=False):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    _, frame = capture.read()
    if gray_flag:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


# Given a frame and additional parameters, display a frame.
def show_frame(frame, bbox_list=None, text=None,
    save_flag=False, save_name=None, wait_flag=False):

    # A list of colors to indicate the order of bounding boxes drawn.
    color_list = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
        [255, 0, 255], [0, 255, 255]]
    color_list = color_list + [255, 255, 255]*20

    # Convert the frame to a BGR image if the input is grayscale.
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # Draw a bounding box, if a bounding box was given.
    if bbox_list:
        for i, bbox in enumerate(bbox_list):
            tl, br = bbox[0], bbox[1]
            frame = cv2.rectangle(frame, tl, br, color_list[i], 4)

    # Draw a text box, if a text string given. Add rectangle to emphasize text.
    if text:
        tbox_tl, tbox_br = (0, 0), (220, 25)
        frame = cv2.rectangle(frame, tbox_tl, tbox_br, (255, 255, 255), -1)

        # Add the text on top of the rectangle to the displayed frame. The
        # cv2.putText() function places text based on the bottom left corner.
        text_bl = (tbox_tl[0] + 5, tbox_br[1] - 5)
        frame = cv2.putText(frame, text, text_bl,
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Display the frame and wait for input if the wait flag is enabled.
    cv2.imshow('frame', frame)
    if wait_flag:
        cv2.waitKey(0)

    # Save the frame if the save_flag is enabled.
    if save_flag:
        cv2.imwrite(save_name, frame)


# Given a list of frame numbers, display each frame. An optional flag exists
# to wait for an input before iterating to the next frame.
def show_frames(capture, frame_num_list, bbox_list=None, wait_flag=True):

    # Iterate through the frame_num_list, while using indexes for bboxes.
    for frame_index, frame_num in enumerate(frame_num_list):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        _, frame = capture.read()
        if bbox_list:
            frame = cv2.rectangle(frame, bbox_list[frame_index][0],
                bbox_list[frame_index][1], [0, 0, 255], 6)
        cv2.imshow('frame', frame)
        if wait_flag:
            cv2.waitKey(0)
        else:
            cv2.waitKey(1)


# Given a list of bounding boxes, return the average bounding box.
def get_avg_bbox(bbox_list):
    total_bboxes = 0
    tl_sum, br_sum = (0, 0), (0, 0)

    # Sum the (tl, br) pairs, using Numpy to easily sum tuples. Skip the
    # summation if (-1) is found, since it represents a missing bbox.
    for bbox in bbox_list:
        if bbox != -1:
            total_bboxes += 1
            tl_sum = np.add(tl_sum, bbox[0])
            br_sum = np.add(br_sum, bbox[1])

    # Round the (tl, br) avg to the nearest int and append to the list.
    tl = (int(round(tl_sum[0]/total_bboxes)),
        int(round(tl_sum[1]/total_bboxes)))
    br = (int(round(br_sum[0]/total_bboxes)),
        int(round(br_sum[1]/total_bboxes)))
    return (tl, br)


#### MATRIX MANIPULATION FUNCTIONS #############################################

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
def resize_img(img, img_scale):
    h, w = img.shape[:2]
    h, w = int(h * img_scale), int(w * img_scale)
    return cv2.resize(img, (w, h))


#### LOGGING FUNCTIONS #########################################################

# Display the total time taken of a test procedure.
def display_total_time(start_time, title=""):
    stop_time = time.time() - start_time
    print("\tTotal {:} Time: {:.5f}s".format(title, stop_time))


# Display the total time taken and average FPS of a test procedure.
def display_fps(start_time, frame_count, title=""):
    stop_time = time.time() - start_time
    print("\tTotal {:} Time: {:.2f}s".format(title, stop_time))
    print("\tAverage FPS: {:.2f}".format(frame_count / stop_time))


# Display the FPS of a parameter analysis test using a finite queue. The first
# element of the time list represents the last update time. Abuses the fact
# that lists are pass by reference, so no list needs to be returned.
def display_pa_fps(start_time, time_list, disp_dict, list_size=60):

    curr_time = time.time()
    if not time_list:
        time_list.append(start_time)
    time_list.append(curr_time - start_time)

    # Update the FPS counter every 0.5s
    avg_time = sum(time_list[1:]) / (len(time_list) - 1)
    if curr_time - time_list[0] >= 0.5:
        time_list[0] = curr_time
        print("====\nFPS: {:04d}".format(int(1/avg_time)))
        for key in disp_dict:
            print("{}: {}".format(key, disp_dict[key]))

    # Delete the first half of the time queue if reached max size.
    if len(time_list) == list_size:
        del time_list[1:list_size//2]
