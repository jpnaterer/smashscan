import time
import numpy as np
import cv2

# SmashScan libraries
import util
import timeline

LABELS_LIST = ["battlefield", "dreamland", "finaldest",
               "fountain", "pokemon", "yoshis"]


# An object that takes a capture and a number of input parameters and performs
# a number of object detection operations. Parameters include a cv2 capture,
# darkflow object, save_flag for saving results, and show_flag for display.
class StageDetector:

    def __init__(self, capture, tfnet, show_flag=False, save_flag=False):
        self.capture = capture
        self.tfnet = tfnet
        self.save_flag = save_flag
        self.show_flag = show_flag

        # Predetermined parameters that have been tested to work best.
        self.end_fnum = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.max_num_match_frames = 30
        self.min_match_length_s = 30
        self.num_match_frames = 5
        self.step_size = 60
        self.timeline_empty_thresh = 4


    #### STAGE DETECTOR TESTS ##################################################

    # Run the standard stage detector test over the entire video.
    def standard_test(self):

        # Create a timeline of the label history where the labels are stored as
        # integers while no result is (-1). Also create a bounding box list.
        dirty_timeline, bbox_hist = list(), list()

        # Iterate through video and use tfnet to perform object detection.
        start_time = time.time()
        for fnum in range(0, self.end_fnum, self.step_size):
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, fnum)
            _, frame = self.capture.read()

            # Get the tfnet result with the largest confidence and extract info.
            bbox, label, confidence = self.get_tfnet_result(frame)

            # Store label if result found, or (-1) if no result was found.
            if label:
                dirty_timeline.append(LABELS_LIST.index(label))
                bbox_hist.append(bbox)
            else:
                dirty_timeline.append(-1)
                bbox_hist.append(-1)

            # Display the frame if show_flag is enabled. Exit if q pressed.
            if self.show_flag:
                if confidence:
                    text = '{}: {:.0f}%'.format(label, confidence * 100)
                    util.show_frame(frame, bbox_list=[bbox], text=text,
                        save_flag=self.save_flag, 
                        save_name="output/{:07d}.png".format(fnum))
                else:
                    util.show_frame(frame, save_flag=self.save_flag,
                        save_name="output/{:07d}.png".format(fnum))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # End the TfNet session and display time taken to complete.
        util.display_fps(start_time, len(dirty_timeline), "Initial Sweep")

        # Fill holes in the history timeline list, and filter out timeline
        # sections that are smaller than a particular size.
        clean_timeline = timeline.fill_filter(dirty_timeline,
            self.timeline_empty_thresh)
        clean_timeline = timeline.size_filter(clean_timeline,
            self.step_size, self.min_match_length_s)
        timeline.show_plots(dirty_timeline, clean_timeline, LABELS_LIST)

        # Get a list of the matches and avg bboxes according to clean_timeline.
        match_ranges = timeline.get_ranges(clean_timeline)
        match_bboxes = self.get_match_bboxes(match_ranges, bbox_hist)

        # Show the beginning and end of each match according to the filters.
        display_frames, display_bboxes = list(), list()
        for i, match_range in enumerate(match_ranges):
            display_frames += [match_range[0]*self.step_size,
                match_range[1]*self.step_size]
            display_bboxes += [match_bboxes[i], match_bboxes[i]]
        util.show_frames(self.capture, display_frames, display_bboxes)


    #### STAGE DETECTOR INTERNAL METHODS #######################################

    # Return the tfnet prediction with the highest confidence.
    def get_tfnet_result(self, frame):
        results = self.tfnet.return_predict(frame)
        result = dict()
        bbox, label, confidence = None, None, None

        max_confidence = 0
        for result_iter in results:
            if result_iter["confidence"] > max_confidence:
                result = result_iter
                max_confidence = result_iter["confidence"]

        if result:
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            bbox = (tl, br)
            label = result['label']
            confidence = result['confidence']

        return bbox, label, confidence

    # Given match ranges and bounding box history, return a list of the average
    # bounding box (top left and bottom right coordinate pair) of each match.
    def get_match_bboxes(self, match_ranges, bbox_hist):
        match_bboxes = list()
        for mr in match_ranges:
            avg_bbox = util.get_avg_bbox(bbox_hist[mr[0]: mr[1]])
            match_bboxes.append(avg_bbox)
        return match_bboxes


    # Given a list of predicted labels, determine the label that accurately
    # represents the match. If there is too much variance or no label was found
    # assume that a highlight reel or poor quality stream was given.
    def get_match_label(self, label_list, match_range):

        # If no stages were found, return and declare a failure.
        if not label_list:
            print("\tUnidentifiable Match Range: {}".format(match_range))
            return None

        # If there is too much variance (multiple labels found), return.
        if len(set(label_list)) > 1:
            print("\tRemoved Match Range: {}".format(match_range))
            return "multiple_stages_found"

        # Find the label that occured the most during the tested frames.
        match_label = max(set(label_list), key=label_list.count)
        return match_label


    #### STAGE DETECTOR EXTERNAL METHODS #######################################

    # Given a list of match ranges, randomly select frames from each range and
    # return the average bounding box and expected label for each match. Also
    # return an updated list of the match ranges, with matches removed where
    # multiple stages were found, aka highlight reels.
    def get_match_info(self, match_ranges):

        # For each match range, generated random frame numbers to search.
        new_match_ranges, match_bboxes, match_labels = list(), list(), list()
        for match_range in match_ranges:
            random_fnum_list = np.random.randint(low=match_range[0],
                high=match_range[1], size=self.num_match_frames)
            bbox_list, label_list = list(), list()

            # Find the labels for the random frame numbers selected.
            for random_fnum in random_fnum_list:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, random_fnum)
                _, frame = self.capture.read()
                bbox, label, _ = self.get_tfnet_result(frame)
                if label:
                    bbox_list.append(bbox)
                    label_list.append(label)

            # Attempt to find a stage if none was found in the initial search.
            if not label_list:
                for _ in range(self.max_num_match_frames):
                    fnum = np.random.randint(match_range[0], match_range[1])
                    self.capture.set(cv2.CAP_PROP_POS_FRAMES, fnum)
                    _, frame = self.capture.read()
                    bbox, label, _ = self.get_tfnet_result(frame)
                    if label:
                        bbox_list, label_list = [bbox], [label]
                        break

            # Find the label that occured the most during the tested frames. If
            # zero or multiple stages were found, declare failure.
            match_label = self.get_match_label(label_list, match_range)
            if match_label is None:
                return None, None
            if match_label == "multiple_stages_found":
                continue
            new_match_ranges.append(match_range)
            match_bboxes.append(util.get_avg_bbox(bbox_list))
            match_labels.append(match_label)

        return new_match_ranges, match_bboxes, match_labels
