from cv2 import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
from datetime import datetime

# SmashScan libraries
import preprocess

options = {
    'config': 'cfg',
    'model': 'cfg/tiny-yolo-voc-6c.cfg',
    'metaLoad': 'cfg/tiny-yolo-voc-6c.meta',
    'pbLoad': 'cfg/tiny-yolo-voc-6c.pb',
    'threshold': 0.25,
    'gpu': 1.0
}

labels_list = ["battlefield", "dreamland", "finaldest", 
    "fountain", "pokemon", "yoshis"]

# Display the main test plot.
def show_tfnet_results(video_name, step_size, 
    videos_dir, save_flag, show_flag):

    # Create an OpenCV capture object. https://docs.opencv.org/3.4.2/
    capture = cv2.VideoCapture('%s/%s' % (videos_dir, video_name))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # A list of the label history to be used in the cleaning algorithm. It 
    # stores the labels as integers, while no result found is (-1).
    dirty_hist = list()

    # A list that stores the corresponding bounding boxes of the timeline.
    bbox_hist = list()

    # Display a cv2 window if the hide flag is disabled.
    if show_flag:
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', 1280, 720)

    # Initialize DarkFlow TFNet object with weights from cfg folder.
    start_time = datetime.now()
    tfnet = TFNet(options)

    # Iterate through video and use tfnet to perform object detection.
    # while (current_frame < total_frames):
    for current_frame in range(0, total_frames, step_size):
        capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        _, frame = capture.read()
        results = tfnet.return_predict(frame)

        # Extract the result with the largest confidence.
        max_confidence = 0
        for result_iter in results:
            if result_iter["confidence"] > max_confidence:
                result = result_iter
                max_confidence = result_iter["confidence"]

        # Extract information from results (list of dicts).
        if max_confidence != 0 and show_flag:
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            confidence = result['confidence']

            # Draw bounding box around frame's result.
            frame = cv2.rectangle(frame, tl, br, [0, 0, 255], 6)

            # Add a white rectangle to the frame to emphasize text.
            text_tl = (tl[0] + 10, tl [1] + 30)
            text_br = (text_tl[0] + 240, text_tl[1] + 20)
            frame = cv2.rectangle(frame, tl, text_br, (255, 255, 255), -1)

            # Add text with label and confidence to the displayed frame.
            text = '{}: {:.0f}%'.format(label, confidence * 100)
            frame = cv2.putText(frame, text, text_tl, 
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 2)

        # Store label if result found, or (-1) if no result was found.
        if max_confidence != 0:
            dirty_hist.append(labels_list.index(result['label']))
            bbox_hist.append((tl, br))
        else:
            dirty_hist.append(-1)
            bbox_hist.append(-1)

        # Display the frame if show_flag is enabled. Close if 'q' pressed.
        if show_flag:
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Save the frame if the save_flag is enabled.
        if save_flag:
            cv2.imwrite('output/frame%07d.png' % current_frame, frame)

    # End the TfNet session and display time taken to complete.
    tfnet.sess.close()
    finish_time = (datetime.now() - start_time).total_seconds()
    print("Successfully collected tfnet results in %.2fs" % finish_time)
    print("\tAverage FPS: %.2f" % (len(dirty_hist) / finish_time))

    # Fill holes in the history timeline list, and filter out timeline 
    # sections that are smaller than a particular size.
    clean_hist = preprocess.hist_fill_filter(dirty_hist)
    clean_hist = preprocess.hist_size_filter(clean_hist, step_size)
    show_hist_plots(dirty_hist, clean_hist, labels_list)
    print(preprocess.get_avg_bboxes(clean_hist, bbox_hist))

    # Show the beginning and end of each match according to the filters.
    match_ranges = preprocess.get_match_ranges(clean_hist)
    for match_range in match_ranges:
        capture.set(cv2.CAP_PROP_POS_FRAMES, match_range[0]*step_size)
        _, frame = capture.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(0)

        capture.set(cv2.CAP_PROP_POS_FRAMES, match_range[1]*step_size)
        _, frame = capture.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(0)

    capture.release()
    cv2.destroyAllWindows()


# Display the dirty and clean history plots. Each plot is
# associated with a dict of lists that contain the labelled
# history of what was returned by the DarkNet object.
def show_hist_plots(dirty_hist, clean_hist, y_labels):
    # Create a figure with two plots (dirty and clean)
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    fig.canvas.set_window_title("History Plots")

    # Setup dirty history scatter plot.
    ax1.scatter(range(len(dirty_hist)), dirty_hist)
    ax1.yaxis.set_ticks(range(len(y_labels)))
    ax1.yaxis.set_ticklabels(y_labels, range(len(y_labels)))
    ax1.set_xlim([-1, len(dirty_hist)])
    ax1.set_ylim([-0.5, len(y_labels) - 0.5])

    # Setup cleaned history scatter plot.
    ax2.scatter(range(len(clean_hist)), clean_hist)
    ax2.yaxis.set_ticks(range(len(y_labels)))
    ax2.yaxis.set_ticklabels(y_labels, range(len(y_labels)))
    ax2.set_xlim([-1, len(dirty_hist)])
    ax2.set_ylim([-0.5, len(y_labels) - 0.5])

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A testing tool used to \
        analyze the performance of trained DarkNet weights.')
    parser.add_argument('video_name', type=str, 
        help='The name of the video file to be tested on.')
    parser.add_argument('-save', '--save_flag', action='store_true',
        help='A flag used to determine if frames are saved.') 
    parser.add_argument('-hide', '--hide_flag', action='store_true',
        help='A flag used to hide the plot, so testing runs faster.') 
    parser.add_argument('-step', '--step_size', type=int, default=60, 
        nargs='?', help='The step size used when testing.')   
    parser.add_argument('-dir', '--video_dir', type=str, default='videos', 
        nargs='?', help='The video file directory to be used.')
    
    args = parser.parse_args()

    show_tfnet_results(args.video_name, args.step_size, 
        args.video_dir, args.save_flag, not args.hide_flag)
