import time
import cv2

# SmashScan Libraries
import percent_matching
import stage_detection
import util

# An object that takes a video location and a number of input parameters and
# performs a number of video content analysis operations.
class VideoAnalyzer:

    def __init__(self, video_location, tfnet, show_flag=False):
        print("Video Location: {:}".format(video_location))
        self.capture = cv2.VideoCapture(video_location)
        self.sd = stage_detection.StageDetector(
            self.capture, tfnet, show_flag=show_flag)
        self.pm = percent_matching.PercentMatcher(
            self.capture, show_flag=show_flag)


    def standard_test(self):

        # Use the Percent Matcher to get an estimate of the match ranges.
        start_time = time.time()
        match_ranges = self.pm.get_match_ranges()
        util.display_total_time(start_time, "Percent Matching")

        # Use the Stage Detector to determine the match bboxes and labels and
        # exit if a stage was not found within any of the match ranges.
        start_time = time.time()
        new_match_ranges, match_bboxes, match_labels = \
            self.sd.get_match_info(match_ranges)
        if match_labels:
            util.display_total_time(start_time, "Stage Detection")
            print("\tMatch Bboxes: {:}".format(match_bboxes))
            print("\tMatch Labels: {:}".format(match_labels))
            match_times = [(mr[1]-mr[0])//30 for mr in new_match_ranges]
            print("\tMatch Times: {:}".format(match_times))
        else:
            print("\tNo stage detected!")
            return False

        # Use the Percent Matcher to get an estimate of the ports in use.
        start_time = time.time()
        port_nums = self.pm.get_port_num_list(new_match_ranges, match_bboxes)
        util.display_total_time(start_time, "Port Sweep")
        print("\tPorts in Use: {:}".format(port_nums))

        # Create a dict to return the relevant video info.
        video_info = dict()
        video_info['match_times'] = match_times
        video_info['match_labels'] = match_labels
        video_info['port_nums'] = port_nums
        return video_info
