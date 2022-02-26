def K_Means(dist, k):
# K-Means algorithm: For a given set of data points, we randomly choose k centers. According to 
# the distance, each data point chooses the closest center. Then, we obtain k clusters. For each
# cluster, find the center with the minimum sumation of distances to other points in the cluster.
# The algorithm stops until the centers are stable.
#
# If the number of clusters are unknown, we can use the elbow method
#
# dist: The matrix to record the distance between any pair of data points.
# k: The number of clusters 
    center = []
    n = len(dist) # Number of nodes
    
    while (len(center) < k):
        index = random.randint(0, n-1)
        if (index not in center):
            center.append(index)
    
    while (True):
        # Determine the center for each data point
        cluster = [[] for _ in range(k)]
        for i in range(n):
            min_cluster = -1
            min_dist = 1000000
            for j in range(k):
                if (dist[i][center[j]] < min_dist):
                    min_dist = dist[i][center[j]]
                    min_cluster = j
            cluster[min_cluster].append(i)
        
        # Select the new centers
        old_center = center.deepcopy()
        center = []
        for i in range(k):
            c = -1
            min_tot_d = 10000
            for j in range(len(cluster[i])):
                tot_d = 0
                for t in range(len(cluster[i])):
                    tot_d += dist[cluster[i][j]][cluster[i][t]]
                if (tot_d < min_tot_d):
                    min_tot_d = tot_d
                    c = cluster[i][j]
            center.append(c)
        if (center.sort() == old_center.sort()):
            break
    
    cls = [-1 for _ in range(n)] # the cluster index for each data point
    for i in range(k):
        for j in range(len(cluster[i])):
            cls[cluster[i][j]] = i
    
    return cls


# DBSCAN: Density-Based Spatial Clustering of Applications with Noise. 
# Finds core samples of high density and expands clusters from them. 
# Good for data which contains clusters of similar density.
# 
# Input:
# dist: the matrix to record distance between any pair of data points
# radius: the radius for the close neighbors
# k: the threshold to determine core points
#
# Core Point: the data point who has at least k neighbors withn the radius
# Boundary Point: the data point within the radius of a core point but it is not a core point
# Noise Point: the data point is not core point neither boundary point.

def core_Point(dist, p, k, radius):
# Check whether the point p is a core point
    s = 0
    for i in range(len(dist)):
        if (dist[p][i] <= radius):
            s += 1
    return (s >= k)

def dbSCAN(dist, k, radius):
    n = len(dist)
    cls = []
    for i in range(n):
        cls.append(-1-i)
    cur = -1
    for i in range(n):
        # Find a core point
        if (cls[i] > -1):
            continue
        new_add = []
        sat = 0
        for j in range(n):
            if (dist[i][j] <= radius):
                sat += 1
        if (sat >= k):
            cur += 1
            cls[i] = cur
        
        # Add data points to the current cluster using Queues data structure
        pointer = 0
        while (len(new_add) > pointer):
            if (core_Point(dist, new_add[pointer], k, radius) == False):
                pointer += 1
                continue
            x = new_add[pointer]
            for j in range(n):
                if (cls[j] > -1):
                    continue
                if (dist[x][j] <= radius):
                    cls[j] = cur
                    new_add.append(j)
            pointer += 1
    return cls


# Conformal DBSCAN: Add the conformal prediction framework to DBSCAN method. The radius is adjusted
# in the iterations.
# 
# Core Points: the data points, whose non-conformity scores are larger than 0.9, in the current cluster
# 
# Input:
# cur: The index of the current cluster
# dist: the matrix to record the distance between any pair of data points
# k: A parameter used in the conformal prediction framework to adjust the radius
# ratio: the threshold for the non-conformity score
# cls: A list to record the cluster index for data points


def conformal_Cluster(cur, dist, k, ratio, cls):
    n = len(dist)
    total = []

    # Choose the initial seeds from the unclustered data points
    for i in range(n):
        if (cls[i] > -1):
            continue
        temp = []
        for j in range(n):
            if (cls[j] > -1):
                continue
            temp.append(dist[i][j])
        temp = sorted(temp)
        if (len(temp) < k+2):
            return cls
        total.append([i, sum(temp[:k]), temp])
    if (len(total) == 0):
        return cls
    total = sorted(total, key = lambda x: x[1])
    radius = 1
    for i in range(k - 1, 0, -1):
        if total[0][2][i] < 1:
            radius = total[0][2][i]
            break
    cur += 1
    core = [total[0][0]]

    # Add new data points
    new_add = [total[0][0]]
    cls[total[0][0]] = cur
    f = 0
    temp_ratio = ratio
    while (True):
        print("Core: ", core, radius, len(core))
        f = False
        # include all the unclusted data points within the radius of core points
        for i in range(n):
            if (cls[i] > -1):
                continue
            for j in core:
                if (dist[i][j] <= radius):
                    cls[i] = cur
                    new_add.append(i)
                    f = True
                    break
        if (f == False):
            break

        # Select the core points
        total = []
        core = []
        for i in range(len(new_add)):
            temp = []
            for j in range(len(new_add)):
                temp.append(dist[new_add[i]][new_add[j]])
            temp = sorted(temp)
            total.append([new_add[i], sum(temp[:int(k)])])
        total = sorted(total, key = lambda x:x[1])
        for i in range(int(0.9*len(new_add))):
            core.append(total[i][0])
        
        # Adjust the radius
        for i in range(len(core)):
            temp = []
            for j in range(len(core)):
                temp.append(dist[core[i]][core[j]])
            temp = sorted(temp)
            if (len(temp) <= k//2):
                continue
            if (temp[k//2-1] > radius):
                radius = temp[k//2-1]

    print("Cluster: ", len(new_add), new_add)
    cls = conformal_Cluster(cur, dist, k, ratio, cls)
    return cls
        
