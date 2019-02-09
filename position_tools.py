import numpy as np

# Given a list of x coordinates that the percent template was found, cluster
# the positions, and calculate the modes of the clusters to determine the most
# accurate position of each port.
def get_port_pos_list(x_pos_list, cluster_diff=40, min_cluster_count=5):

    # Sort the unique x-positions to make clustering easier. Append a large
    # number to the list to ensure that the final cluster is accounted for.
    x_cluster_list, prev_cluster_start = list(), 0
    x_pos_list_sorted = np.sort(np.unique(x_pos_list)).tolist()
    x_pos_list_sorted = x_pos_list_sorted + [1000]

    # Iterate through the sorted position list, and create a new cluster if the
    # difference between two adjacent positions is greater than a threshold.
    for i in range(0, len(x_pos_list_sorted)-1):
        if x_pos_list_sorted[i+1] - x_pos_list_sorted[i] > cluster_diff:
            x_cluster_list.append(
                x_pos_list_sorted[prev_cluster_start:i+1])
            prev_cluster_start = i+1

    # Decide a port position exists if it is the most occuring value within the
    # cluster, and it occurs more than a min count.
    x_cluster_list_mode = list()
    for x_cluster in x_cluster_list:
        current_count = min_cluster_count
        current_x = -1
        for x in x_cluster:
            if x_pos_list.count(x) > current_count:
                current_count = x_pos_list.count(x)
                current_x = x
        if current_x != -1:
            x_cluster_list_mode.append(current_x)

    return x_cluster_list_mode


# Given a list of port x-positions and the associated match bounding box,
# determine the port numbers that are in use using rough positional estimates.
def get_port_num_list(port_pos_list, match_bbox):

    # Subtract the match left-coor to get a position within the melee frame.
    port_pos_list_adj = [x - match_bbox[0][0] for x in port_pos_list]
    ports_used = list()

    # Use rough positional estimates to determine which ports exist.
    for x in port_pos_list_adj:
        if 0 <= x < 150:
            ports_used.append(1)
        elif 150 <= x < 250:
            ports_used.append(2)
        elif 250 <= x < 350:
            ports_used.append(3)
        elif 350 <= x < 450:
            ports_used.append(4)
    return ports_used
