
import numpy as np
import matplotlib.pyplot as plt

# functions from the previous post
def distance_matrix(arr1, arr2):
    """
    In this function, we compute the Euclidean distance matrix between two sequences.

    We use two for loops to fill distance values into the cells for illustration reason.

    For a more compact form, simply use
    distance = [ [ np.sqrt( (arr1[i] - arr2[j])**2 ) for i in np.arange(len(arr1))] for j in np.arange(len(arr2))]

    """

    # construct an empty array
    distance = np.ndarray(shape = (len(arr1), len(arr2)))

    for row in np.arange(len(arr1)):
        for col  in np.arange(len(arr2)):
            distance[row,col] = np.sqrt( (arr1[row] - arr2[col])**2 )

    return distance


def dtw_basic(arr1, arr2, alignment_curve = False, gen_plot = True):
    """
    This function shows a basic DTW calculation with two multi-dimensional arrays.

    Notice that we modified the outputs a litte, it outputs (dtw_distance, dtw_matrix) now.
    """

    # initialize the dtw array
    dtw = np.zeros(shape = (len(arr1), len(arr2)))

    for row in range(dtw.shape[0]):
        for col in range(dtw.shape[1]):
            # calculate distance between arr1[row] and arr2[col]
            # here we use Euclidean distance
            dist = np.sqrt( np.sum( (arr1[row] - arr2[col])**2 ) )

            # the starting point
            if row == 0 and col == 0:
                dtw[row, col] = dist
            # we can only go right along the upmost row
            elif row == 0:
                dtw[row, col] = dist + dtw[row, col - 1]
            # we can only go down along the leftmost column
            elif col == 0:
                dtw[row, col] = dist + dtw[row-1, col]
            # the recursive relation
            else:
                dtw[row, col] = dist + min(dtw[row-1, col], dtw[row-1, col-1], dtw[row, col-1])

    # alignment curve
    if alignment_curve:

        row = 0
        col = 0
        alignment = [ [row, col] ]

        while row != dtw.shape[0] - 1 or col != dtw.shape[1] - 1:
            if row == dtw.shape[0] - 1:
                col += 1
            elif col == dtw.shape[1] - 1:
                row += 1
            else:
                idx = np.argsort( [ dtw[row+1, col], dtw[row+1, col+1], dtw[row, col+1] ]  )[0]
                if idx == 0:
                    row += 1
                elif  idx == 1:
                    row += 1
                    col += 1
                else:
                    col += 1
            alignment.append([row, col])

        alignment = np.array(map(np.array, alignment))

    if gen_plot:
        # plotting
        fig = plt.figure(figsize = (5,5))
        plt.imshow( dtw )
        plt.xlim(-0.1, dtw.shape[1] - 1)
        plt.ylim(dtw.shape[0] - 1, -0.1)
        plt.title("Basic Dynamic Time Warping Matrix Heat Map")
        if alignment_curve:
            plt.plot( alignment[:,1], alignment[:,0], linewidth = 3, color = 'white', label = 'alignment curve')
            plt.legend(loc = 'best')
        plt.show()

    if alignment_curve:
        return dtw, alignment
    else:
        return dtw

def dtw_constraint(template, unknown, window = 10, alignment_curve = False, gen_plot = False):

    template = np.array(template)
    unknown = np.array(unknown)
    # local constraint
    window = max( window, abs(len(template)-len(unknown)) )
    # dtw matrix initialization
    dtw = np.ndarray(shape = (len(template), len(unknown)))
    dtw[:,:] = float('inf')

    for row in range(dtw.shape[0]):
        # set the constraint
        for col in range(max(0, row-window), min(len(unknown), row+window)):
            dist = np.sqrt( np.sum( (template[row] - unknown[col])**2 ) )
            if row == 0 and col == 0:
                dtw[row, col] = dist
            elif row == 0:
                dtw[row, col] = dist + dtw[row, col - 1]
            elif col == 0:
                dtw[row, col] = dist + dtw[row-1, col]
            else:
                dtw[row, col] = dist + min(dtw[row-1, col], dtw[row-1, col-1], dtw[row, col-1])
                idx = np.argsort( [ dtw[row-1, col], dtw[row-1, col-1], dtw[row, col-1] ]  )[0]

    # trace for alignment curve
    if alignment_curve:
        row = 0
        col = 0
        alignment = [ [row, col] ]

        while row != dtw.shape[0] - 1 or col != dtw.shape[1] - 1:
            if row == dtw.shape[0] - 1:
                col += 1
            elif col == dtw.shape[1] - 1:
                row += 1
            else:
                idx = np.argsort( [ dtw[row+1, col], dtw[row+1, col+1], dtw[row, col+1] ]  )[0]
                if idx == 0:
                    row += 1
                elif  idx == 1:
                    row += 1
                    col += 1
                else:
                    col += 1
            alignment.append([row, col])
        alignment = np.array(map(np.array, alignment))

    if gen_plot:
        fig = plt.figure(figsize = (7,7))
        plt.imshow( dtw )
        plt.xlim(0, dtw.shape[1])
        plt.ylim(0, dtw.shape[0])
        plt.title("Constrained Dynamic Time Warping Matrix Heat Map")
        if alignment_curve:
            plt.plot( alignment[:,1], alignment[:,0], linewidth = 3, color = 'white', label = 'alignment curve')
            plt.legend(loc = 'best')
        plt.show()


    if alignment_curve:
        return dtw, alignment
    else:
        return dtw

def dtw_plot(query, template, alignment_curve):
    """
    Function to show how two sequneces are connected by DTW.
    """
    fig = plt.figure(figsize = (14,7))
    # shift the query sequence up
    plt.plot(np.arange(len(query)), query + 2, lw = 2, label = 'query')
    plt.plot(np.arange(len(template)), template, lw = 2, label = 'template')
    plt.legend(loc = 'best')
    for x1, x2 in alignment_curve:
        #print query[x1]+2, a['seq_%s' %0][x2]
        plt.plot([x1,x2], [template[x1], query[x2]+2], 'r')

def z_normalize(array):

    return 1.0*(array - np.mean(array))/np.std(array)
