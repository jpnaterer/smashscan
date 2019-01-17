import time
import numpy as np
import cv2
from darkflow.net.build import TFNet

# SmashScan libraries
import util
import timeline

TFNET_OPTIONS = {
    'config': 'cfg',
    'model': 'cfg/tiny-yolo-voc-6c.cfg',
    'metaLoad': 'cfg/tiny-yolo-voc-6c.meta',
    'pbLoad': 'cfg/tiny-yolo-voc-6c.pb',
    'threshold': 0.25,
    'gpu': 1.0
}

LABELS_LIST = ["battlefield", "dreamland", "finaldest",
               "fountain", "pokemon", "yoshis"]

# Display the main tfnet results.
def show_tfnet_results(video_location, step_size, save_flag, show_flag):

    # Create an OpenCV capture object. https://docs.opencv.org/3.4.2/
    capture = cv2.VideoCapture(video_location)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create a timeline of the label history where the labels are stored as
    # integers while no result is (-1). Also create a bounding box list.
    dirty_timeline = list()
    bbox_hist = list()

    # Initialize DarkFlow TFNet object with weights from cfg folder.
    start_time = time.time()
    tfnet = TFNet(TFNET_OPTIONS)

    # Iterate through video and use tfnet to perform object detection.
    for current_frame in range(0, total_frames, step_size):
        capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        _, frame = capture.read()

        # Get the tfnet result with the largest confidence and extract info.
        result = get_tfnet_result(tfnet, frame)
        if result:
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            confidence = result['confidence']

        # Store label if result found, or (-1) if no result was found.
        if result:
            dirty_timeline.append(LABELS_LIST.index(result['label']))
            bbox_hist.append((tl, br))
        else:
            dirty_timeline.append(-1)
            bbox_hist.append(-1)

        # Display the frame if show_flag is enabled. Add a bounding box, and
        # label+confidence string if a result was found. Exit if 'q' is pressed.
        if show_flag:
            if result:
                util.show_frame(frame, bbox_list=[(tl, br)],
                    text='{}: {:.0f}%'.format(label, confidence * 100))
            else:
                cv2.imshow('frame', frame)

            # Exit the video iteration if 'q' is pressed.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Save the frame if the save_flag is enabled.
        if save_flag:
            cv2.imwrite('output/frame%07d.png' % current_frame, frame)

    # End the TfNet session and display time taken to complete.
    finish_time = time.time() - start_time
    print("Initial video sweep tfnet results in %.2fs" % finish_time)
    print("\tAverage FPS: %.2f" % (len(dirty_timeline) / finish_time))

    # Fill holes in the history timeline list, and filter out timeline
    # sections that are smaller than a particular size.
    clean_timeline = timeline.fill_filter(dirty_timeline)
    clean_timeline = timeline.size_filter(clean_timeline, step_size)
    timeline.show_plots(dirty_timeline, clean_timeline, LABELS_LIST)

    # Get a list of the matches and avg bboxes according to clean_timeline.
    match_ranges = timeline.get_match_ranges(clean_timeline)
    match_bboxes = get_match_bboxes(match_ranges, bbox_hist)

    # Show the beginning and end of each match according to the filters.
    display_frames = list()
    display_bboxes = list()
    for i, match_range in enumerate(match_ranges):
        display_frames += [match_range[0]*step_size, match_range[1]*step_size]
        display_bboxes += [match_bboxes[i], match_bboxes[i]]
    util.show_frames(capture, display_frames, display_bboxes)

    tfnet.sess.close()
    capture.release()
    cv2.destroyAllWindows()


# Return the tfnet prediction with the highest confidence.
def get_tfnet_result(tfnet, frame):
    results = tfnet.return_predict(frame)
    result = dict()

    max_confidence = 0
    for result_iter in results:
        if result_iter["confidence"] > max_confidence:
            result = result_iter
            max_confidence = result_iter["confidence"]
    return result


# Given the match ranges and bounding box history, return a list of the average
# bounding box (top left and bottom right coordinate pair) of each match.
def get_match_bboxes(match_ranges, bbox_hist):
    match_bboxes = list()

    # Iterate through match ranges and initialize match_size to be the counter
    # for the number of elements summed, and tl_sum/br_sum as the actual sums.
    for match_range in match_ranges:
        match_size = 0
        tl_sum, br_sum = (0, 0), (0, 0)

        # Use the match_range to iterate through the bbox_hist to sum the
        # (tl, br) pair. Numpy is used to easily sum the tuples.
        for i in range(match_range[0], match_range[1]):
            if bbox_hist[i] != -1:
                match_size += 1
                tl_sum = np.add(tl_sum, bbox_hist[i][0])
                br_sum = np.add(br_sum, bbox_hist[i][1])

        # Round the (tl, br) avg to the nearest int and append to the list.
        tl = (int(round(tl_sum[0]/match_size)),
              int(round(tl_sum[1]/match_size)))
        br = (int(round(br_sum[0]/match_size)),
              int(round(br_sum[1]/match_size)))
        match_bboxes.append((tl, br))

    return match_bboxes
