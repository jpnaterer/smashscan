import argparse
import time
import os
import cv2
from darkflow.net.build import TFNet

# SmashScan libraries
import percent_matching
import stage_detection
import thresholding
import util
import video_analysis

# Darkflow/Tensorflow default settings:
TFNET_OPTIONS = {
    'config': 'cfg',
    'model': 'cfg/tiny-yolo-voc-6c.cfg',
    'metaLoad': 'cfg/tiny-yolo-voc-6c.meta',
    'pbLoad': 'cfg/tiny-yolo-voc-6c.pb',
    'threshold': 0.25,
    'gpu': 1.0
}


# Run the PM test over a wide range of input parameters.
def run_all_pm_tests(test_type_str, video_location,
    start_fnum, stop_fnum, save_flag, show_flag, wait_flag):

    # Create a capture object and set the stop frame number if none was given.
    capture = cv2.VideoCapture(video_location)
    if stop_fnum == 0:
        stop_fnum = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Run the PM test with grayscale and non-grayscale parameters.
    for gray_flag in [True, False]:

        # Display the flags used for the current PM test.
        print("==== Percent Matching Test ====")
        print("\tgray_flag={}".format(gray_flag))
        print("\tshow_flag={}".format(show_flag))
        pm = percent_matching.PercentMatcher(capture, [start_fnum, stop_fnum],
            gray_flag, save_flag, show_flag, wait_flag)

        # Run the PM test according to the input test_type_str.
        if test_type_str == "pms":
            pm.sweep_test()
        elif test_type_str == "pmc":
            pm.calibrate_test()
        elif test_type_str == "pmi":
            pm.initialize_test()
        elif test_type_str == "pmt":
            pm.timeline_test()

    # Release the OpenCV capture object.
    capture.release()


# Run the VA test over the entire video folder.
def run_va_over_video_folder():

    # Create a single darkflow object to be reused by each video analyzer.
    start_total_time = time.time()
    tfnet = TFNet(TFNET_OPTIONS)
    video_info_list = list()

    # Perform video analysis over all the videos in the video directory.
    for file_name in os.listdir('videos'):

        # Skip the files that do not contain a .mp4 extension.
        if "mp4" not in file_name:
            continue
        video_location = "{:s}/{:s}".format('videos', file_name)
        va = video_analysis.VideoAnalyzer(video_location,
            tfnet, args.show_flag)
        video_info_list.append(va.standard_test())

    # Calculate the overall video statistics
    print("====TOTAL STATS====")
    match_times, match_labels, match_ports = list(), list(), list()
    for video_info in video_info_list:
        match_times.extend(video_info['match_times'])
        match_labels.extend(video_info['match_labels'])
        match_ports.extend([j for i in video_info['match_ports'] for j in i])

    # Calculate time statistics.
    avg_time = sum(match_times)//len(match_times)
    print("Average match time {:}:{:02d}".format(avg_time//60, avg_time%60))

    # Calculate stage statistics.
    label_list = ["battlefield", "dreamland", "finaldest",
        "fountain", "pokemon", "yoshis"]
    for label in label_list:
        label_percentage = 100*match_labels.count(label)/len(match_labels)
        print("Stage {:}: {:.2f}%".format(label, label_percentage))

    # Calculate port number statistics.
    for i in [1, 2, 3, 4]:
        port_percentage = 100*match_ports.count(i)/len(match_ports)
        print("Port {:}: {:.2f}%".format(i, port_percentage))

    util.display_total_time(start_total_time, "Analyze")


if __name__ == '__main__':
    # Create a CLI parser and add a video file positional argument.
    parser = argparse.ArgumentParser(description='A testing tool used to \
        analyze the performance of trained DarkNet weights.')
    parser.add_argument('video_name', type=str,
        help='The name of the video file to be tested on.')

    # Add CLI arguments to run various smashscan tests.
    parser.add_argument('-pms', '--pms_test_flag', action='store_true',
        help='A flag used to run the percent matching sweep test.')
    parser.add_argument('-pmc', '--pmc_test_flag', action='store_true',
        help='A flag used to run the percent matching calibrate test.')
    parser.add_argument('-pmi', '--pmi_test_flag', action='store_true',
        help='A flag used to run the percent matching initialize test.')
    parser.add_argument('-pmt', '--pmt_test_flag', action='store_true',
        help='A flag used to run the percent matching timeline test.')
    parser.add_argument('-sdt', '--sdt_test_flag', action='store_true',
        help='A flag used to run the stage detection timeline test.')
    parser.add_argument('-vaf', '--vaf_test_flag', action='store_true',
        help='A flag used to run the video analyzer folder test.')
    parser.add_argument('-pah', '--pah_test_flag', action='store_true',
        help='A flag used to run the HSV parameter analysis test.')
    parser.add_argument('-pad', '--pad_test_flag', action='store_true',
        help='A flag used to run the damage parameter analysis test.')

    # Add CLI arguments for parameters of the various smashscan tests.
    parser.add_argument('-show', '--show_flag', action='store_true',
        help='A flag used to display the results as each test runs.')
    parser.add_argument('-wait', '--wait_flag', action='store_true',
        help='A flag used to wait for key inputs during displaying frames.')
    parser.add_argument('-save', '--save_flag', action='store_true',
        help='A flag used to determine if frames are saved.')
    parser.add_argument('-start', '--start_fnum', type=int, default=0,
        nargs='?', help='The initial frame to begin testing.')
    parser.add_argument('-stop', '--stop_fnum', type=int, default=0,
        nargs='?', help='The final frame to end testing.')

    # Parse the CLI arguments and create a compact video location string.
    args = parser.parse_args()
    video_location = "{:s}/{:s}".format('videos', args.video_name)

    # Run the smashscan test indicated by the input flags (tfnet by default).
    if args.pms_test_flag:
        run_all_pm_tests("pms", video_location, args.start_fnum,
            args.stop_fnum, args.save_flag, args.show_flag, args.wait_flag)
    elif args.pmc_test_flag:
        run_all_pm_tests("pmc", video_location, args.start_fnum,
            args.stop_fnum, args.save_flag, args.show_flag, args.wait_flag)
    elif args.pmi_test_flag:
        run_all_pm_tests("pmi", video_location, args.start_fnum,
            args.stop_fnum, args.save_flag, args.show_flag, args.wait_flag)
    elif args.pmt_test_flag:
        run_all_pm_tests("pmt", video_location, args.start_fnum,
            args.stop_fnum, args.save_flag, args.show_flag, args.wait_flag)
    elif args.sdt_test_flag:
        capture = cv2.VideoCapture(video_location)
        tfnet = TFNet(TFNET_OPTIONS)
        sd = stage_detection.StageDetector(capture, tfnet,
            args.show_flag, args.save_flag)
        sd.standard_test()
    elif args.vaf_test_flag:
        run_va_over_video_folder()
    elif args.pah_test_flag:
        pah = thresholding.HsvParamAnalyzer(video_location, args.start_fnum)
        pah.standard_test()
    elif args.pad_test_flag:
        pad = thresholding.DmgParamAnalyzer(video_location,
            args.start_fnum, args.stop_fnum)
        pad.standard_test()
    else:
        tfnet = TFNet(TFNET_OPTIONS)
        va = video_analysis.VideoAnalyzer(video_location,
            tfnet, args.show_flag)
        va.standard_test()
