import numpy as np

# note: the /etc/matplotlibrc (on Ubuntu 16)  file has to be configured to use
# the 'Agg' backend (see the file for details).  This is so it will work
# in the PostgreSql environment

def format_coord(x, y):
    col = int(x + 0.5)
    row = int(y + 0.5)
    if col >= 0 and col < numcols and row >= 0 and row < numrows:
        z = X[row, col]
        return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
    else:
        return 'x=%1.4f, y=%1.4f' % (x, y)


def unpack(z):
    retval = []
    for row in z:
        row1 = []
        row2 = []
        for e in row:
            row1.append(e[0])
            row1.append(e[1])
            row2.append(e[2])
            row2.append(e[3])
        retval.append(row1)
        retval.append(row2)
    return retval

def plot(x_ticks, y_ticks, z_data, file_name, title):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    unp = unpack(z_data)
    
    fig, ax = plt.subplots()
    fig.suptitle(title, fontsize=8)
    X = np.array(unp)
    ax.imshow(X, cmap=cm.hot, interpolation='nearest')

    numrows, numcols = X.shape
    numrows /= 2
    numcols /=2

    ax.format_coord = format_coord

    plt.xticks(np.arange(len(x_ticks)), tuple(x_ticks), rotation='vertical')

    plt.yticks(np.arange(len(y_ticks)), tuple(y_ticks), rotation='horizontal')

    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.tick_params(axis='both', which='minor', labelsize=7)

    plt.tight_layout()

    plt.savefig(file_name)
