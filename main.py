import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    import h5py
    filename = "train_catvnoncat.h5"

    with h5py.File(filename, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

        # Get the data
        data = list(f[a_group_key])
        datay = f['train_set_y']
        datay = np.array(datay)

        datax = f['train_set_x']
        datax = np.array(datax)

    index = 25

    print(datay.shape)
    print(datay[25])

    print(datax.shape)
    print(datax[25])

    plt.imshow(datax[25])
    plt.show()
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
