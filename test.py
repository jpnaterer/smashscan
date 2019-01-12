import time
import argparse
import cv2
from darkflow.net.build import TFNet

# SmashScan libraries
import preprocess
import util
import template_matcher

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
    start_time = time.time()
    tfnet = TFNet(TFNET_OPTIONS)

    # Iterate through video and use tfnet to perform object detection.
    for current_frame in range(0, total_frames, step_size):
        capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        _, frame = capture.read()

        # Get the tfnet result with the largest confidence and extract info.
        result = preprocess.get_tfnet_result(tfnet, frame)
        if result:
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            confidence = result['confidence']

        # Store label if result found, or (-1) if no result was found.
        if result:
            dirty_hist.append(LABELS_LIST.index(result['label']))
            bbox_hist.append((tl, br))
        else:
            dirty_hist.append(-1)
            bbox_hist.append(-1)

        # Display the frame if show_flag is enabled. Add a bounding box, and
        # label+confidence string if a result was found. Exit if 'q' is pressed.
        if show_flag:
            if result:
                util.show_frame(frame, bbox=[tl, br],
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
    print("\tAverage FPS: %.2f" % (len(dirty_hist) / finish_time))

    # Fill holes in the history timeline list, and filter out timeline
    # sections that are smaller than a particular size.
    clean_hist = preprocess.hist_fill_filter(dirty_hist)
    clean_hist = preprocess.hist_size_filter(clean_hist, step_size)
    preprocess.show_hist_plots(dirty_hist, clean_hist, LABELS_LIST)

    # Get a list of the matches and avg bboxes according to clean_hist.
    match_ranges = preprocess.get_match_ranges(clean_hist)
    match_bboxes = preprocess.get_match_bboxes(match_ranges, bbox_hist)

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


# Run the TM test over a wide range of input parameters.
def run_all_tm_tests(video_location, step_size, start_fnum, show_flag):
    # Create an OpenCV capture object. https://docs.opencv.org/3.4.2/
    capture = cv2.VideoCapture(video_location)

    # Run the TM test over various parameter configurations,
    run_tm_test(capture, step_size, start_fnum, gray_flag=False,
        roi_flag=False, show_flag=show_flag)
    run_tm_test(capture, step_size, start_fnum, gray_flag=True,
        roi_flag=False, show_flag=show_flag)
    run_tm_test(capture, step_size, start_fnum, gray_flag=False,
        roi_flag=True, show_flag=show_flag)
    run_tm_test(capture, step_size, start_fnum, gray_flag=True,
        roi_flag=True, show_flag=show_flag)

    # Release the OpenCV capture object.
    capture.release()


# Run a single TM test over a given group of input parameters.
def run_tm_test(capture, step_size, start_fnum, 
    gray_flag, roi_flag, show_flag):

    # Define the range of frames to be scanned during the test.
    test_range = 3000
    end_fnum = start_fnum + test_range

    # Start a timer and complete the TM test.
    start_time = time.time()
    template_matcher.tm_test(capture, step_size,
        frame_range=[start_fnum, end_fnum], gray_flag=gray_flag,
        roi_flag=roi_flag, show_flag=show_flag)
    finish_time = time.time() - start_time
    average_fps = (test_range // step_size) / finish_time

    # Display the flags used and the time taken to complete the test.
    print("==== Template Matching Test ====")
    print("\tgray_flag={}".format(gray_flag))
    print("\troi_flag={}".format(roi_flag))
    print("\tshow_flag={}".format(show_flag))
    print("\tTotal time: {:.2f}s".format(finish_time))
    print("\tAverage FPS: {:.2f}".format(average_fps))


if __name__ == '__main__':
    # Create a CLI parser and add a video file positional argument.
    parser = argparse.ArgumentParser(description='A testing tool used to \
        analyze the performance of trained DarkNet weights.')
    parser.add_argument('video_name', type=str,
        help='The name of the video file to be tested on.')

    # Add a number of keyword arguments for various testing parameters.
    parser.add_argument('-save', '--save_flag', action='store_true',
        help='A flag used to determine if frames are saved.')
    parser.add_argument('-hide', '--hide_flag', action='store_true',
        help='A flag used to hide the plot, so testing runs faster.')
    parser.add_argument('-step', '--step_size', type=int, default=60,
        nargs='?', help='The step size used when testing.')
    parser.add_argument('-dir', '--video_dir', type=str, default='videos',
        nargs='?', help='The video file directory to be used.')

    # Add CLI arguments to run various smashscan tests.
    parser.add_argument('-tm_test', '--tm_test_flag', action='store_true',
        help='A flag used to run the template matching test.')
    parser.add_argument('-show', '--show_flag', action='store_true',
        help='A flag used to display the results as each test runs.')
    parser.add_argument('-start', '--start_fnum', type=int, default=0,
        nargs='?', help='The initial frame to begin testing.')

    # Parse the CLI arguments and create a compact video location string.
    args = parser.parse_args()
    video_location = "{:s}/{:s}".format(args.video_dir, args.video_name)

    # Run the smashscan test indicated by the input flags (tfnet by default).
    if args.tm_test_flag:
        run_all_tm_tests(video_location, args.step_size, 
            args.start_fnum, args.show_flag)
    else:
        show_tfnet_results(video_location, args.step_size,
            args.save_flag, not args.hide_flag)
