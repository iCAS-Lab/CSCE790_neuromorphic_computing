import numpy as np
import cv2 as cv
import tensorflow as tf
import matplotlib.pyplot as plt
################################################################################
TESTING = True
################################################################################


def plot_signal(timesteps, values, filename):
    """
    Plot the signal in matplotlib. If filename is set then save figure.
    """
    fig = plt.figure(figsize=(12, 3))
    ax = fig.add_subplot()
    plt.plot(timesteps, values)
    if filename:
        assert '.' in filename, 'Error --> File extension not specified.'
        plt.savefig(filename)


def rand_signal(value_range=10, timestep_range=1, length=100,
                constant=True, negatives=False):
    """
    Takes in an upper bound for the y values of the signal and an upper bound
    of the timesteps between each value. If constant is set true then the 
    timestep between each point is equal. 

    This function creates a random signal of size length. It
    outputs timesteps of the signal and the values at those timesteps.
    """
    timesteps = []
    values = []
    total_time = 0
    for i in range(0, length):
        if constant:
            timesteps.append(i*timestep_range)
            values.append(np.random.randint(
                negatives*(-value_range), value_range))
        else:
            assert not timestep_range == 1, 'ERROR --> timestep_range must be > 1.'
            rand_time = np.random.randint(1, timestep_range+1)
            total_time += rand_time
            timesteps.append(total_time)
            values.append(np.random.randint(
                negatives*(-value_range), value_range))
        if total_time == length or total_time + timestep_range > length:
            break
    return (np.asarray(timesteps), np.asarray(values))


def rand_smooth_signal():
    """
    Like rand_signal creates a random signal which is smooth.
    """
    timesteps = []
    values = []
    return (timesteps, values)


def regular_signal():
    """
    Creats a signal that oscilates at a recuring frequency for a duration.
    """
    timesteps = []
    values = []
    return (timesteps, values)


def cos_signal():
    """
    Generates a cosine signal for a specified duration.
    """
    timesteps = []
    values = []
    return (timesteps, values)


def sin_signal():
    """
    Generates a sine signal for a specified duration.
    """
    timesteps = []
    values = []
    return (timesteps, values)


def edge_signal(image):
    """
    From
    P. Chandarana, J. Ou and R. Zand, "An Adaptive Sampling and Edge Detection
    Approach for Encoding Static Images for Spiking Neural Networks," 2021 12th
    International Green and Sustainable Computing Conference (IGSC), 2021, pp. 
    1-8, doi: 10.1109/IGSC54211.2021.9651610.

    Takes in a 2D list or a 2D numpy array.

    Creates two signals, an x and y signal, which identify which pixels are on
    an edge in the image.

    Timestep is by default 1.
    """
    assert type(image) == np.ndarray, \
        "Error --> Input image not a numpy ndarray."
    edge_image = cv.Canny(image, 100, 200)
    if np.argmax(edge_image) > 1:
        edge_image = edge_image/255.0
    y, x = np.where(edge_image == 1)
    return (x, y)
################################################################################
################################################################################
################################################################################


def testing():
    """
    Just a function for testing the functions
    """
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    mnist_sample = x_train[1]
    x, y = edge_signal(mnist_sample)
    print((x, y))
    plt.imsave('mnist.png', mnist_sample)
    plot_signal(np.arange(1, len(x)+1), x, 'mnist_x.png')
    plot_signal(np.arange(1, len(y)+1), y, 'mnist_y.png')
    plot_signal(y, x, 'combined_x_y.png')
    # Test Random Signal Generation
    t1, v1 = rand_signal(timestep_range=3, constant=False)
    plot_signal(t1, v1, 'test1.png')


if TESTING:
    testing()
