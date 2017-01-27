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


def plot(x_ticks, y_ticks, z_data, file_name, title, labsize):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    fig, ax = plt.subplots()
    #fig.suptitle(title, fontsize=8)
    X = np.array(z_data)

    ax.imshow(X, cmap=cm.hot, interpolation='nearest')

    numrows, numcols = X.shape
    ax.format_coord = format_coord

    plt.xticks(np.arange(len(x_ticks)), tuple(x_ticks), rotation='vertical')

    plt.yticks(np.arange(len(y_ticks)), tuple(y_ticks), rotation='horizontal')

    ax.tick_params(axis='both', which='major', labelsize=labsize)
    ax.tick_params(axis='both', which='minor', labelsize=labsize)
    # ax.set_xticklabels(xticklabels, fontsize=6)
    # ax.set_xticklabels(xticklabels, fontsize=6)
    # ax.set_yticklabels(yticklabels, fontsize=6)
    #plt.xticks(np.arange(6), ('a', 'b', 'c', 'd', 'e', 'f'))

    plt.tight_layout()

    plt.savefig(file_name)
    #plt.show()

    #pl.plot(x_ticks, y_ticks, z_data, fname)
