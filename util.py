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
def show_frame(frame, bbox_list=None, text=None, wait_flag=False):

    # A list of colors to indicate the order of bounding boxes drawn.
    color_list = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]]

    # Convert the frame to a BGR image if the input is grayscale.
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # Draw a bounding box, if a bounding box was given.
    if bbox_list:
        for i, bbox in enumerate(bbox_list):
            tl, br = bbox[0], bbox[1]
            frame = cv2.rectangle(frame, tl, br, color_list[i], 4)

    # Draw a text box, if a text string was given.
    if text:
        # Add a white rectangle to the frame to emphasize text.
        tbox_tl = (0, 0)
        tbox_br = (tbox_tl[0] + 220, tbox_tl[1] + 25)
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


#### LOGGING FUNCTIONS #########################################################

# Display the total time taken of a test procedure.
def display_total_time(start_time, title=""):
    stop_time = time.time() - start_time
    print("\tTotal {:} Time: {:.2f}s".format(title, stop_time))


# Display the total time taken and average FPS of a test procedure.
def display_fps(start_time, frame_count, title=""):
    stop_time = time.time() - start_time
    print("\tTotal {:} Time: {:.2f}s".format(title, stop_time))
    print("\tAverage FPS: {:.2f}".format(frame_count / stop_time))
