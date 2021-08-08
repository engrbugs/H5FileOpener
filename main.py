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
        classes = f['list_classes']
        train_set_x = f['train_set_x']
        train_set_y = f['train_set_y']
        classes = np.array(classes)
        train_set_x = np.array(train_set_x)
        train_set_y = np.array(train_set_y)

    f.close()
    # Example of a picture
    index = 1
    plt.imshow(train_set_x[index])
    print("y = " + str(train_set_y[index]) + ", it's a '" + classes[np.squeeze(train_set_y[index])].decode(
        "utf-8") + "' picture.")
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
