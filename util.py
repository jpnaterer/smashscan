import cv2
import numpy as np

# Given a frame and additional parameters, display a frame.
def show_frame(frame, bbox=None, text=None):

    # Draw a bounding box, if a bounding box was given.
    if bbox:
        tl, br = bbox[0], bbox[1]
        frame = cv2.rectangle(frame, tl, br, [0, 0, 255], 4)

    # Draw a text box, if a bounding box and text string was given.
    if bbox and text:

        # Add a white rectangle to the frame to emphasize text.
        tbox_tl = (0, 0)
        tbox_br = (tbox_tl[0] + 240, tbox_tl[1] + 30)
        frame = cv2.rectangle(frame, tbox_tl, tbox_br, (255, 255, 255), -1)

        # Add the text on top of the rectangle to the displayed frame.
        text_bl = (tbox_tl[0] + 5, tbox_br[1] - 5)
        frame = cv2.putText(frame, text, text_bl,
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

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


# Given a template location, extract the image and alpha (transparent) mask.
def get_image_and_mask(image_location, resize_ratio=None, gray_flag=False):

    # Load image from file with alpha channel (UNCHANGED flag).
    img = cv2.imread(image_location, cv2.IMREAD_UNCHANGED)

    # If alpha channel exists, extract it. Otherwise, return just the image.
    if img.shape[2] > 3:
        # Create a alpha channel matrix  with values between 0-255.
        channels = cv2.split(img)
        mask = np.array(channels[3])

        # Threshold the alpha channel to create a binary mask.
        _,mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Convert image and mask to grayscale or BGR based on input flag.
        if gray_flag:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Resize the image and mask based on input value.
        if resize_ratio:
            h, w = img.shape[:2]
            h = int(h * resize_ratio)
            w = int(w * resize_ratio)
            img = cv2.resize(img, (w, h))
            mask = cv2.resize(mask, (w, h))

        return img, mask

    else:
        return img, None
