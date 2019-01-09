import cv2

# Given a frame and additional parameters, display a frame.
def show_frame(frame, bbox=None, text=None):

    # Draw a bounding box, if a bounding box was given.
    if bbox:
        tl, br = bbox[0], bbox[1]
        frame = cv2.rectangle(frame, tl, br, [0, 0, 255], 6)

    # Draw a text box, if a bounding box and text string was given.
    if bbox and text:

        # Add a white rectangle to the frame to emphasize text.
        text_tl = (tl[0] + 10, tl [1] + 30)
        text_br = (text_tl[0] + 240, text_tl[1] + 20)
        frame = cv2.rectangle(frame, tl, text_br, (255, 255, 255), -1)

        # Add the text on top of the rectangle to the displayed frame.
        frame = cv2.putText(frame, text, text_tl,
            cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 2)

    # Display the frame and return.
    cv2.imshow('frame', frame)


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
