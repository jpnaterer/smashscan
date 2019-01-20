import time
import argparse
import cv2

# SmashScan libraries
import stage_detector
import percent_matcher


# Run the TM test over a wide range of input parameters.
def run_all_tm_tests(test_type_str, video_location,
    step_size, start_fnum, stop_fnum, num_frames, show_flag, wait_flag):

    # Create a capture object and set the stop frame number if none was given.
    capture = cv2.VideoCapture(video_location)
    if stop_fnum == 0:
        stop_fnum = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Run the TM test over various parameter configurations,
    run_tm_test(capture, test_type_str, step_size, start_fnum, stop_fnum,
        num_frames, show_flag, wait_flag, gray_flag=False)
    run_tm_test(capture, test_type_str, step_size, start_fnum, stop_fnum,
        num_frames, show_flag, wait_flag, gray_flag=True)

    # Release the OpenCV capture object.
    capture.release()


# Run a single TM test over a given group of input parameters.
def run_tm_test(capture, test_type_str, step_size, start_fnum, stop_fnum,
    num_frames, show_flag, wait_flag, gray_flag):

    # Initialize the TM object.
    tm = percent_matcher.TemplateMatcher(capture, step_size,
        [start_fnum, stop_fnum], num_frames, gray_flag, show_flag, wait_flag)

    # Display the flags used for the current TM test.
    print("==== Template Matching Test ====")
    print("\tgray_flag={}".format(gray_flag))
    print("\tshow_flag={}".format(show_flag))

    # Run TM initialization if the test requires it.
    if test_type_str == "tmt":
        start_time = time.time()
        tm.initialize_template_scale()
        print("\tTotal Init Time: {:.2f}s".format(time.time() - start_time))

    # Run the TM test according to the input test_type_str and end the timer.
    start_time = time.time()
    if test_type_str == "tms":
        tm.standard_test()
    elif test_type_str == "tmc":
        tm.calibrate_test()
    elif test_type_str == "tmi":
        tm.initialize_test()
    elif test_type_str == "tmt":
        tm.timeline_test()


if __name__ == '__main__':
    # Create a CLI parser and add a video file positional argument.
    parser = argparse.ArgumentParser(description='A testing tool used to \
        analyze the performance of trained DarkNet weights.')
    parser.add_argument('video_name', type=str,
        help='The name of the video file to be tested on.')

    # Add a number of keyword arguments for various testing parameters.
    parser.add_argument('-save', '--save_flag', action='store_true',
        help='A flag used to determine if frames are saved.')
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
    parser.add_argument('-tmt', '--tmt_test_flag', action='store_true',
        help='A flag used to run the template matching timeline test.')

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
    elif args.tmt_test_flag:
        run_all_tm_tests("tmt", video_location, args.step_size,
            args.start_fnum, args.stop_fnum, args.num_frames,
            args.show_flag, args.wait_flag)
    else:
        capture = cv2.VideoCapture(video_location)
        sd = stage_detector.StageDetector(capture, args.step_size,
            args.save_flag, args.show_flag)
        sd.standard_test()
