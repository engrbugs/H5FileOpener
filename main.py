import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):

    filename = "test_catvnoncat.h5"

    with h5py.File(filename, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        # Get the data
        header = list(f.keys())
        classes = f['list_classes']
        train_set_x = f['test_set_x']
        train_set_y = f['test_set_y']
        classes = np.array(classes)
        train_set_x = np.array(train_set_x)
        train_set_y = np.array(train_set_y)
        f.close()
    # Example of a picture
    index = 25
    plt.imshow(train_set_x[index])
    print("y = " + str(train_set_y[index]) + ", it's a '" + classes[np.squeeze(train_set_y[index])].decode(
        "utf-8") + "' picture.")
    print("total number of " + str(header[1]) + ": " + str(train_set_x.shape[0]))
    print("total number of " + str(header[2]) + ": " + str(train_set_y.shape[0]))
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
