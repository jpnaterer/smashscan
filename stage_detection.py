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


# An object that takes a capture and a number of input parameters and performs
# a number of object detection operations. Parameters include a step_size for
# the speed of iteration, save_flag for saving results, and show_flag to
# display results.
class StageDetector:

    def __init__(self, capture, show_flag=False, save_flag=False):
        self.capture = capture
        self.save_flag = save_flag
        self.show_flag = show_flag

        # Predetermined parameters that have been tested to work best.
        self.end_fnum = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.min_match_length_s = 30
        self.num_match_frames = 5
        self.step_size = 60
        self.timeline_empty_thresh = 4

        # Initialize DarkFlow TFNet object with weights from cfg folder.
        self.tfnet = TFNet(TFNET_OPTIONS)


    def __del__(self):
        self.tfnet.sess.close()


    #### STAGE DETECTOR TESTS ##################################################

    # Run the standard stage detector test over the entire video.
    def standard_test(self):

        # Create a timeline of the label history where the labels are stored as
        # integers while no result is (-1). Also create a bounding box list.
        dirty_timeline, bbox_hist = list(), list()

        # Iterate through video and use tfnet to perform object detection.
        start_time = time.time()
        for current_frame in range(0, self.end_fnum, self.step_size):
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
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
                    util.show_frame(frame, bbox_list=[bbox], text=text)
                else:
                    util.show_frame(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Save the frame if the save_flag is enabled.
            if self.save_flag:
                cv2.imwrite('output/frame%07d.png' % current_frame, frame)

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


    #### STAGE DETECTOR EXTERNAL METHODS #######################################

    # Given a list of match ranges, randomly select frames from each range and
    # return the average bounding box and expected label for each match.
    def get_match_data(self, match_ranges):

        # For each match range, generated random frame numbers to search.
        match_bboxes, match_labels = list(), list()
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

            # Find the label that occured the most during the tested frames.
            match_label = max(set(label_list), key=label_list.count)
            match_bboxes.append(util.get_avg_bbox(bbox_list))
            match_labels.append(match_label)

        return match_bboxes, match_labels
