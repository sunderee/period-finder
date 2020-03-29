import numpy as np
from scipy.io import loadmat
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import operator
import statistics


def plot_dataset(input_matrix: np.ndarray, x_axis_size: int):
    """
    Plot the dataset stored in the matrix Y... We know that there are 10
    signals, each having 512 samples, therefore we can define the X axis
    ourselves.
    :param input_matrix: input Numpy matrix
    :param x_axis_size: size of X axis
    :return: None
    """
    x_axis = [x for x in range(1, x_axis_size + 1)]
    figure, axes = plt.subplots(5, 2)
    axes[0, 0].plot(x_axis, input_matrix[0])
    axes[1, 0].plot(x_axis, input_matrix[1])
    axes[2, 0].plot(x_axis, input_matrix[2])
    axes[3, 0].plot(x_axis, input_matrix[3])
    axes[4, 0].plot(x_axis, input_matrix[4])
    axes[0, 1].plot(x_axis, input_matrix[5])
    axes[1, 1].plot(x_axis, input_matrix[6])
    axes[2, 1].plot(x_axis, input_matrix[7])
    axes[3, 1].plot(x_axis, input_matrix[8])
    axes[4, 1].plot(x_axis, input_matrix[9])
    plt.show()


def compute_autocorrelation(input_signal: np.ndarray) -> tuple:
    """
    Compute autocorrelation from the given discrete signal
    :param input_signal: the discrete signal we would like to process
    :return: array of autocorrelation results
    """
    size = input_signal.size
    norm = input_signal - np.mean(input_signal)
    correlation = np.correlate(norm, norm, mode='same')
    return correlation[size // 2 + 1:] / \
           (input_signal.var() * np.arange(size - 1, size // 2, -1))


def compute_period(input_autocorrelations: np.ndarray) -> float:
    """
    Compute period from autocorrelation data by finding most prominent
    peaks and determining if the distance between them is similar. Values
    used in the find_peaks function were chosen arbitrarily based on data
    from autocorrelation plots: height of prominent peaks was always above
    0.1, distance between them was larger than 10 and each peak should
    have a width of 1 sample.
    :param input_autocorrelations: input Numpy array of autocorrelations
    :return: result of the period, which should be a non-negative number
             if period is found, and -1.0 if the results are inconclusive
    """
    peaks, _ = find_peaks(input_autocorrelations, height=0.1, distance=10, width=1)
    if len(peaks) >= 2:
        periods = list(map(operator.sub, peaks[1:], peaks[:-1]))
        if statistics.stdev(periods) < 1.0:
            return np.average(periods)
        else:
            return -1.0
    else:
        return -1.0


# Load the dataset and retrieve matrix Y as Numpy array, but you
# need to transpose it from 512x10 to 10x512 for easier analysis
dataset = loadmat('../assets/DN2Data.mat')
matrix_y = np.array(dataset['Y']).transpose()

# Plot signals from the dataset
plot_dataset(matrix_y, 512)

# Compute autocorrelations and plot them
autocorrelation_list = []
for i in range(10):
    autocorrelation_list.append(compute_autocorrelation(matrix_y[i]))
plot_dataset(np.array(autocorrelation_list), 255)

# Compute periods for all signals (if this is possible)
period_list = []
for autocorr in autocorrelation_list:
    period_list.append(compute_period(autocorr))

# Filter through periods to find valid signals
for i, period in enumerate(period_list):
    if period != -1.0:
        print('Valid signal: index ' + str(i + 1) + ', period ' + str(int(period)) + ' samples')
