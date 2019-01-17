import matplotlib.pyplot as plt

# A custom filter that works on an int array that represents the timeline of
# labels (stages) found by tfnet. (-1) represents that no stage was found. The
# goal of this filter is to fill in small time segment holes, while also
# filtering out small time segments.
def fill_filter(dirty_timeline_in, differ_thresh=4):
    # Add some no-stage found states at the end of dirty_timeline to allow the
    # filter defined below work at the end of the list. This fix is necessary
    # when the match ends too close (with differ_thresh) to the end of the
    # video. These states will be removed after the filtering is complete.
    # The function input is not directly modified since arrays are mutable.
    dirty_timeline = dirty_timeline_in + [-1]*differ_thresh

    # Assume that the history timeline has no stages present (-1).
    clean_timeline = [-1] * len(dirty_timeline)

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

    # Iterate through dirty_timeline and perform the filtering defined above.
    for i, stored_state in enumerate(dirty_timeline):
        if current_state != stored_state:
            if differ_state == stored_state:
                differ_count += 1
                differ_const_count += 1
            else:
                differ_count += 1
                differ_const_count = 1
                differ_state = stored_state

            if differ_const_count == differ_thresh:
                differ_count = 0
                differ_const_count = 0
                current_state = differ_state
                clean_timeline[i-(differ_thresh-1):i] = \
                    [current_state] * (differ_thresh-1)
            elif differ_count == differ_thresh and current_state != -1:
                differ_count = 0
                current_state = -1
                clean_timeline[i-(differ_thresh-1):i] = \
                    [current_state] * (differ_thresh-1)
        else:
            differ_count = 0
            differ_const_count = 0
            differ_state = stored_state
        clean_timeline[i] = current_state

    # Remove the no-stage states inserted at the input of the filter.
    dirty_timeline = dirty_timeline[:-differ_thresh]
    clean_timeline = clean_timeline[:-differ_thresh]

    return clean_timeline


# A custom filter that works on an int array that represents the timeline of
# labels found by tfnet. (-1) represents that no stage was found. The goal of
# this filter is to remove all time segments shorter than match_length_thresh.
def size_filter(dirty_timeline, step_size):
    # The time required for a time segment to be considered gameplay. Assumes
    # the game is captured as 30fps, and the minimum match length is 30s.
    match_length_thresh = int(30 * (30 / step_size))

    # Filter out matches that are less than match_length_thresh.
    match_ranges = get_match_ranges(dirty_timeline)
    match_ranges = list(filter(
        lambda b: b[1] - b[0] > match_length_thresh, match_ranges))

    # Assume that the history timeline has no stages present (-1). Then
    # update the original clean_timeline timeline by removing short matches.
    clean_timeline = [-1] * len(dirty_timeline)
    for match_range in match_ranges:
        clean_timeline[match_range[0]:match_range[1]] = \
            [dirty_timeline[match_range[0]]] * (match_range[1] - match_range[0])

    return clean_timeline


# Given a label timeline, return a list of pairs corresponding to
# the ranges (starting and ending frames) a match (!= -1) is present.
def get_match_ranges(any_timeline):
    match_ranges = list()

    # Indicates the current stage while iterating through the timeline.
    current_state = -1

    # Indicates the timestep the current stage was first detected.
    start_timestep = 0

    # The algorithm requires a stage transition at the end of the timeline.
    used_timeline = any_timeline + [-1]

    # Iterate through the timeline. A match start is indicated by a change
    # from a (-1) to non-(-1) state, while a match end is indicated by a
    # change from a non-(-1) to (-1) state.
    for i, stored_state in enumerate(used_timeline):
        if stored_state != -1 and current_state == -1:
            current_state = stored_state
            start_timestep = i
        elif stored_state != current_state and current_state != -1:
            current_state = -1
            match_ranges.append((start_timestep, i - 1))

    return match_ranges


# Display the dirty and clean timeline plots. The timeline has a y-range
# associated with the various labels that each timeline can represent.
def show_plots(dirty_timeline, clean_timeline, y_labels):
    # Create a figure with two plots (dirty and clean)
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    fig.canvas.set_window_title("History Plots")

    # Setup dirty history scatter plot.
    ax1.scatter(range(len(dirty_timeline)), dirty_timeline)
    ax1.yaxis.set_ticks(range(len(y_labels)))
    ax1.yaxis.set_ticklabels(y_labels, range(len(y_labels)))
    ax1.set_xlim([-1, len(dirty_timeline)])
    ax1.set_ylim([-0.5, len(y_labels) - 0.5])

    # Setup cleaned history scatter plot.
    ax2.scatter(range(len(clean_timeline)), clean_timeline)
    ax2.yaxis.set_ticks(range(len(y_labels)))
    ax2.yaxis.set_ticklabels(y_labels, range(len(y_labels)))
    ax2.set_xlim([-1, len(dirty_timeline)])
    ax2.set_ylim([-0.5, len(y_labels) - 0.5])

    plt.show()
