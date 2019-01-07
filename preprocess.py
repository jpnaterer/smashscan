import numpy as np
import cv2
import matplotlib.pyplot as plt

# Return the tfnet prediction with the highest confidence.
def get_tfnet_result(tfnet, frame):
    results = tfnet.return_predict(frame)
    result = dict()

    max_confidence = 0
    for result_iter in results:
        if result_iter["confidence"] > max_confidence:
            result = result_iter
            max_confidence = result_iter["confidence"]
    return result


# A custom filter that works on an int array that represents the timeline of 
# labels (stages) found by tfnet. (-1) represents that no stage was found. The 
# goal of this filter is to fill in small time segment holes, while also 
# filtering out small time segments. 
def hist_fill_filter(dirty_hist_in, differ_thresh=4):
    # Add some no-stage found states at the end of dirty_hist to allow the
    # filter defined below work at the end of the list. This fix is necessary
    # when the match ends too close (with differ_thresh) to the end of the 
    # video. These states will be removed after the filtering is complete. 
    # The function input is not directly modified since arrays are mutable.
    dirty_hist = dirty_hist_in + [-1]*differ_thresh

    # Assume that the history timeline has no stages present (-1).
    clean_hist = [-1] * len(dirty_hist)

    # Used to store the stage (state) currently present. It will remain the
    # current_state unless differ_thresh number of timesteps differ in a row.
    current_state = -1

    # Used to store the stage (state) that most recently differed from the 
    # current state. current_state will become differ_state if differ_thresh
    # number of timestamps are consistent in a row.
    differ_state = -1

    # The counter used to count the number of times the timestep differs. If
    # the timestep differs but not consistently (different from differ_state),
    # current_state will become the no-stage found state (-1).
    differ_count = 0

    # The counter used to count the number of times the timestep differs. 
    # However, if the timestep consistently differs to differ_state, 
    # current_state will become differ_state once differ_thresh is met.
    differ_const_count = 0

    # Iterate through dirty_hist and perform the filtering defined above.
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
                differ_count = 0
                differ_const_count = 0
                current_state = differ_state
                clean_hist[i-(differ_thresh-1):i] = \
                    [current_state] * (differ_thresh-1)
            elif differ_count == differ_thresh and current_state != -1:
                differ_count = 0
                current_state = -1
                clean_hist[i-(differ_thresh-1):i] = \
                    [current_state] * (differ_thresh-1)
        else:
            differ_count = 0
            differ_const_count = 0
            differ_state = dirty_hist[i]
        clean_hist[i] = current_state

    # Remove the no-stage states inserted at the input of the filter.
    dirty_hist = dirty_hist[:-differ_thresh]
    clean_hist = clean_hist[:-differ_thresh]

    return clean_hist


# A custom filter that works on an int array that represents the timeline of 
# labels found by tfnet. (-1) represents that no stage was found. The goal of
# this filter is to remove all time segments shorter than match_length_thresh.
def hist_size_filter(dirty_hist, step_size):
    # The time required for a time segment to be considered gameplay. Assumes
    # the game is captured as 30fps, and the minimum match length is 30s.
    match_length_thresh = int(30 * (30 / step_size))

    # Assume that the history timeline has no stages present (-1).
    clean_hist = [-1] * len(dirty_hist)

    # Filter out matches that are less than match_length_thresh.
    match_ranges = get_match_ranges(dirty_hist)
    match_ranges = list(filter( 
        lambda b: b[1] - b[0] > match_length_thresh, match_ranges))

    # Update the original clean_hist timeline by removing short matches.
    for match_range in match_ranges:
        clean_hist[match_range[0]:match_range[1]] = \
            [dirty_hist[match_range[0]]] * (match_range[1] - match_range[0])

    return clean_hist


# Given a label timeline, return a list of pairs corresponding to 
# the ranges (starting and ending frames) a match (!= -1) is present.
def get_match_ranges(any_hist):
    match_ranges = list()

    # Indicates the current stage while iterating through the timeline.
    current_state = -1

    # Indicates the timestep the current stage was first detected.
    start_timestep = 0

    # The algorithm requires a stage transition at the end of the timeline.
    used_hist = any_hist + [-1]

    # Iterate through the timeline. A match start is indicated by a change
    # from a (-1) to non-(-1) state, while a match end is indicated by a
    # change from a non-(-1) to (-1) state.
    for i in range(0, len(used_hist)):
        if used_hist[i] != -1 and current_state == -1:
            current_state = used_hist[i]
            start_timestep = i
        elif used_hist[i] != current_state and current_state != -1:
            current_state = -1
            match_ranges.append((start_timestep, i - 1))

    return match_ranges


# Given the match ranges and bounding box history, return a list of the average
# bounding box (top left and bottom right coordinate pair) of each match.
def get_match_bboxes(match_ranges, bbox_hist):
    match_bboxes = list()

    # Iterate through match ranges and initialize match_size to be the counter
    # for the number of elements summed, and tl_sum/br_sum as the actual sums.
    for match_range in match_ranges:
        match_size = 0
        tl_sum, br_sum = (0, 0), (0, 0)

        # Use the match_range to iterate through the bbox_hist to sum the 
        # (tl, br) pair. Numpy is used to easily sum the tuples.
        for i in range(match_range[0], match_range[1]):
            if bbox_hist[i] != -1:
                match_size += 1
                tl_sum = np.add(tl_sum, bbox_hist[i][0])
                br_sum = np.add(br_sum, bbox_hist[i][1])

        # Round the (tl, br) avg to the nearest int and append to the list.
        tl = (int(round(tl_sum[0]/match_size)),
              int(round(tl_sum[1]/match_size)))
        br = (int(round(br_sum[0]/match_size)),
              int(round(br_sum[1]/match_size)))
        match_bboxes.append((tl, br))

    return match_bboxes


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


# Uses tfnet to obtain a more accurate location of match start and end.
def get_accurate_match_ranges(init_match_ranges, orig_step_size,
    capture, tfnet, accurate_threshold=0.5):

    # The final match range list to be returned.
    final_match_ranges = list()

    # The list of step sizes to be used for efficient video traversal.
    ss_index = 0
    step_size_list = [60, 29, 14, 6, 2, 1]
    step_size = step_size_list[ss_index]

    # Iterate through the final match ranges.
    for match_range in init_match_ranges:

        # Find accurate location of start of match.
        start_prediction = match_range[0]*orig_step_size + 1
        ss_index = 0
        while True:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_prediction - step_size)
            _, frame = capture.read()
            result = get_tfnet_result(tfnet, frame)

            if result:
                if result['confidence'] > accurate_threshold:
                    start_prediction -= step_size
                    continue

            if step_size == 1:
                break
            else:
                ss_index += 1
                step_size = step_size_list[ss_index]

        # Find accurate location of end of match.
        end_prediction = match_range[1]*orig_step_size + 1
        ss_index = 0
        while True:
            capture.set(cv2.CAP_PROP_POS_FRAMES, end_prediction + step_size)
            _, frame = capture.read()
            result = get_tfnet_result(tfnet, frame)

            if result:
                if result['confidence'] > accurate_threshold:
                    end_prediction += step_size
                    continue

            if step_size == 1:
                break
            else:
                ss_index += 1
                step_size = step_size_list[ss_index]

        final_match_ranges.append([start_prediction, end_prediction])

    return final_match_ranges
