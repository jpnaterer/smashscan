import time
import argparse
import cv2

# SmashScan libraries
import stage_detector
import template_matcher


# Run the TM test over a wide range of input parameters.
def run_all_tm_tests(test_type_str, video_location,
    step_size, start_fnum, stop_fnum, num_frames, show_flag, wait_flag):

    # Create an OpenCV capture object. https://docs.opencv.org/3.4.2/
    capture = cv2.VideoCapture(video_location)

    # Run the TM test over various parameter configurations,
    run_tm_test(capture, test_type_str, step_size, start_fnum, stop_fnum,
        num_frames, show_flag, wait_flag, gray_flag=False, roi_flag=False)
    run_tm_test(capture, test_type_str, step_size, start_fnum, stop_fnum,
        num_frames, show_flag, wait_flag, gray_flag=True, roi_flag=False)
    run_tm_test(capture, test_type_str, step_size, start_fnum, stop_fnum,
        num_frames, show_flag, wait_flag, gray_flag=False, roi_flag=True)
    run_tm_test(capture, test_type_str, step_size, start_fnum, stop_fnum,
        num_frames, show_flag, wait_flag, gray_flag=True, roi_flag=True)

    # Release the OpenCV capture object.
    capture.release()


# Run a single TM test over a given group of input parameters.
def run_tm_test(capture, test_type_str, step_size, start_fnum, stop_fnum,
    num_frames, show_flag, wait_flag, gray_flag, roi_flag):

    # Start a timer and initialize the TM object.
    start_time = time.time()
    tm = template_matcher.TemplateMatcher(capture, step_size,
        [start_fnum, stop_fnum], num_frames, gray_flag,
        roi_flag, show_flag, wait_flag)

    # Run the TM test according to the input test_type_str and end the timer.
    if test_type_str == "tms":
        tm.standard_test()
        num_frames_tested = (stop_fnum - start_fnum) // step_size
    elif test_type_str == "tmc":
        tm.calibrate_test()
        num_frames_tested = (stop_fnum - start_fnum) // step_size
    elif test_type_str == "tmi":
        tm.initialize_test()
        num_frames_tested = num_frames
    finish_time = time.time() - start_time
    average_fps = num_frames_tested / finish_time

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
    parser.add_argument('-tms', '--tms_test_flag', action='store_true',
        help='A flag used to run the template matching standard test.')
    parser.add_argument('-tmc', '--tmc_test_flag', action='store_true',
        help='A flag used to run the template matching calibrate test.')
    parser.add_argument('-tmi', '--tmi_test_flag', action='store_true',
        help='A flag used to run the template matching initialize test.')

    # Add CLI arguments for parameters of the various smashscan tests.
    parser.add_argument('-show', '--show_flag', action='store_true',
        help='A flag used to display the results as each test runs.')
    parser.add_argument('-wait', '--wait_flag', action='store_true',
        help='A flag used to wait for key inputs during displaying frames.')
    parser.add_argument('-start', '--start_fnum', type=int, default=0,
        nargs='?', help='The initial frame to begin testing.')
    parser.add_argument('-stop', '--stop_fnum', type=int, default=0,
        nargs='?', help='The final frame to end testing.')
    parser.add_argument('-num', '--num_frames', type=int, default=30,
        nargs='?', help='The number of frames used for testing.')

    # Parse the CLI arguments and create a compact video location string.
    args = parser.parse_args()
    video_location = "{:s}/{:s}".format(args.video_dir, args.video_name)

    # Run the smashscan test indicated by the input flags (tfnet by default).
    if args.tms_test_flag:
        run_all_tm_tests("tms", video_location, args.step_size,
            args.start_fnum, args.stop_fnum, args.num_frames,
            args.show_flag, args.wait_flag)
    elif args.tmc_test_flag:
        run_all_tm_tests("tmc", video_location, args.step_size,
            args.start_fnum, args.stop_fnum, args.num_frames,
            args.show_flag, args.wait_flag)
    elif args.tmi_test_flag:
        run_all_tm_tests("tmi", video_location, args.step_size,
            args.start_fnum, args.stop_fnum, args.num_frames,
            args.show_flag, args.wait_flag)
    else:
        stage_detector.show_tfnet_results(video_location, args.step_size,
            args.save_flag, not args.hide_flag)
