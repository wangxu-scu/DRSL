import numpy as np



def wx_calc_map_label(dist, label, k=0, dist_method='L2'):
    ord = dist.argsort()
    numcases = dist.shape[0]
    if k == 0:
        k = numcases
        # k_list = [50, numcases]
    result = []
    # for k in k_list:
    res = []
    for i in range(numcases):
        order = ord[i]
        p = 0.0
        r = 0.0
        for j in range(k):
            if label[i] == label[order[j]]:
                r += 1
                p += (r / (j + 1))
        if r > 0:
            res += [p / r]
        else:
            res += [0]
    return np.mean(res)

