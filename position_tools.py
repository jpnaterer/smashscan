import numpy as np

# Given a list of x coordinates that the percent template was found, cluster
# the positions, and calculate the modes of the clusters to determine the most
# accurate position of each port.
def get_port_pos_list(x_pos_list):
    x_cluster_list, prev_cluster_start = list(), 0
    x_pos_list_sorted = np.sort(np.unique(x_pos_list)).tolist()
    x_pos_list_sorted = x_pos_list_sorted + [1000]
    for j in range(0, len(x_pos_list_sorted)-1):
        if x_pos_list_sorted[j+1] - x_pos_list_sorted[j] > 40:
            x_cluster_list.append(
                x_pos_list_sorted[prev_cluster_start:j+1])
            prev_cluster_start = j+1
    print(x_cluster_list)
    print(x_pos_list)

    x_cluster_list_mode = list()
    for x_cluster in x_cluster_list:
        current_count = 5
        current_x = -1
        for x in x_cluster:
            if x_pos_list.count(x) > current_count:
                current_count = x_pos_list.count(x)
                current_x = x
        if current_x != -1:
            x_cluster_list_mode.append(current_x)
    print(x_cluster_list_mode)
    return x_cluster_list_mode


# Given a list of port x-positions and the associated match bounding box,
# determine the port numbers that are in use using rough positional estimates.
def get_port_num_list(port_pos_list, match_bbox):
    port_pos_list_adj = [x - match_bbox[0][0] for x in port_pos_list]
    print(port_pos_list_adj)
    ports_used = list()
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
