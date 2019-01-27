import time
import cv2

# SmashScan Libraries
import percent_matching
import stage_detection
import util

# An object that takes a video location and a number of input parameters and
# performs a number of video content analysis operations.
class VideoAnalyzer:

    def __init__(self, video_location, show_flag=False):
        self.capture = cv2.VideoCapture(video_location)
        self.sd = stage_detection.StageDetector(
            self.capture, show_flag=show_flag)
        self.pm = percent_matching.PercentMatcher(
            self.capture, show_flag=show_flag)


    def standard_test(self):
        match_ranges = self.pm.timeline_test()

        # Use the Stage Detector to determine the match bboxes and labels.
        start_time = time.time()
        match_bboxes, match_labels = self.sd.get_match_data(match_ranges)
        util.display_total_time(start_time, "Stage Detection")
        print("\tMatch Bboxes: {:}".format(match_bboxes))
        print("\tMatch Labels: {:}".format(match_labels))

        self.pm.port_test(match_ranges, match_bboxes)
