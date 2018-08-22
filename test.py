from cv2 import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
from datetime import datetime

options = {
    'config': 'cfg',
    'model': 'cfg/tiny-yolo-voc-6c.cfg',
    'metaLoad': 'cfg/tiny-yolo-voc-6c.meta',
    'pbLoad': 'cfg/tiny-yolo-voc-6c.pb',
    'threshold': 0.3,
    'gpu': 1.0
}

label_map = {"battlefield": 0, "dreamland": 1, "finaldest": 2, 
    "fountain": 3, "pokemon": 4, "yoshis": 5 }
labels_list = ["battlefield", "dreamland", "finaldest", 
    "fountain", "pokemon", "yoshis"]

# Display the main test plot.
def show_test_plot(video_name, save_flag, step_size, videos_dir, hide_flag):

    # Create an OpenCV capture object. https://docs.opencv.org/3.4.2/
    capture = cv2.VideoCapture('%s/%s' % (videos_dir, video_name))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Display a cv2 window if the hide flag is disabled.
    if not hide_flag:
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', 1280, 720)

    # Initialize DarkFlow TFNet object with weights from cfg folder.
    tfnet = TFNet(options)

    # A list of the label history to be used in the cleaning algorithm. It 
    # stores the labels as integers, while no result found is (-1).
    dirty_hist = list()

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
        if max_confidence != 0 and not hide_flag:
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
                                cv2.FONT_HERSHEY_DUPLEX, 
                                0.8, (0, 0, 0), 2)

        # Store label if result found, or (-1) if no result was found.
        if max_confidence != 0:
            dirty_hist.append(label_map[result['label']])
        else:
            dirty_hist.append(-1)

        # Display the frame if the hide_flag is disabled.
        if not hide_flag:
            cv2.imshow('frame', frame)

        # Save the frame if the save_flag is enabled.
        if save_flag:
            cv2.imwrite('output/frame%07d.png' % current_frame, frame)

        # Stop testing and close the plot if 'q' is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    capture.release()
    cv2.destroyAllWindows()
    tfnet.sess.close()

    # Apply the cleaning algorithm and display a history plot.
    clean_hist = get_clean_hist(dirty_hist)
    show_hist_plots(dirty_hist, clean_hist, labels_list)


# Calculates the cleaned history.
def get_clean_hist(dirty_hist):
    differ_thresh = 5

    clean_hist = [-1] * len(dirty_hist)
    current_state = -1
    differ_state = -1

    differ_count = 0
    differ_const_count = 0

    for i in range(0, len(dirty_hist)):
        if current_state != dirty_hist[i]:
            if differ_state == dirty_hist[i]:
                differ_count += 1
                differ_const_count += 1
            else:
                differ_count += 1
                differ_const_count = 1
                differ_state = dirty_hist[i]

            if differ_const_count == differ_thresh:
                differ_const_count = 0
                current_state = differ_state
                clean_hist[i-(differ_thresh-1):i] = \
                    [current_state] * (differ_thresh-1)
            elif differ_count == differ_thresh:
                differ_count = 0
                current_state = -1
                clean_hist[i-(differ_thresh-1):i] = \
                    [current_state] * (differ_thresh-1)
        else:
            differ_count = 0
            differ_const_count = 0
            differ_state = dirty_hist[i]
        clean_hist[i] = current_state

    # Correction for the end of the video when the stream 
    # transition is too quick for differ_tresh to detect.
    last_state = clean_hist[len(dirty_hist) - 1]
    end_hist = [-1] * (differ_thresh)
    for i in range(len(dirty_hist) - differ_thresh, len(dirty_hist)):
        if clean_hist[i] == dirty_hist[i]:
            end_index = differ_thresh - (len(dirty_hist) - i) + 1
            end_hist[:end_index] = [last_state]*(end_index)
    clean_hist[len(dirty_hist) - differ_thresh:] = end_hist

    #print (dirty_hist)
    #print (clean_hist)
    
    return clean_hist


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
    parser.add_argument('-v', '--video_name', type=str, 
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

    show_test_plot(args.video_name, args.save_flag, 
        args.step_size, args.video_dir, args.hide_flag)
