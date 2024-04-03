from scipy.spatial.distance import euclidean as distance_2p


def seg_length(x, y):
    distance = 0
    for i in range(0, len(x) - 1):
        distance += distance_2p([x[i], y[i]], [x[i + 1], y[i + 1]])
    return distance


def chord_length(x, y):
    return distance_2p([x[0], y[0]], [x[len(x) - 1], y[len(y) - 1]])


def tortuosity(x, y):
    assert len(x) == len(y)
    if len(x) == 1:
        return None
    return seg_length(x, y) / chord_length(x, y)
