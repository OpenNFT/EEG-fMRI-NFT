import numpy as np
from scipy.interpolate import CubicSpline
from stockwell import st
import multiprocessing as mp
import json

json_info = json.load(open('model/bands_info.json', 'r'))
channel = open('model/channel.txt', 'r').read().replace('\n', '')


# Model dependent arrays, depends on your trained model
NORM_MEAN = np.array(json_info[channel]['norm_mean']).reshape(-1, 1)

NORM_SD = np.array(json_info[channel]['norm_sd']).reshape(-1, 1)

bands = np.array(json_info[channel]['bands'])


def init_pool_worker():
    global bands


def cubic_spline_interpolator(signal, source_f, target_f, axis=-1):
    interpolator = CubicSpline(np.arange(0, signal.shape[axis]), signal, axis=axis)
    target_grid = np.arange(0, signal.shape[axis], source_f / target_f)
    return interpolator(target_grid)


def average_into_bands(stock_power, stock_freqs, band_boundaries):
    band_means = []
    band_freqs = []
    if len(stock_power.shape) == 2:
        stock_power = stock_power.reshape(1, stock_power.shape[0], stock_power.shape[1])
    for band_idx in range(len(band_boundaries) - 1):
        band_indexes = (stock_freqs >= band_boundaries[band_idx]) * (stock_freqs < band_boundaries[band_idx + 1])
        selected = stock_power[:, band_indexes, :]
        selected[selected == 0] = selected[selected != 0].min()
        selected = np.log10(selected)
        band_means.append(
            selected.mean(axis=1).reshape([1, 1, -1])
        )
        band_freqs.append(
            stock_freqs[band_indexes].mean()
        )

    band_means = np.concatenate(band_means, axis=1)
    return band_means


def preprocess_data(sample, sample_rate, workers_pool, fmin=0, fmax=60, target_freq=4):
    df = 0.05  # sampling step in frequency domain (Hz)
    fmin_samples = int(fmin / df)
    fmax_samples = int(fmax / df)

    args = [(sample, fmin_samples, fmax_samples, df, bands)]

    transformed_bands = workers_pool.starmap(stockwell_one_channel, args)

    transformed_bands = np.concatenate(transformed_bands, axis=1)
    transformed_bands = cubic_spline_interpolator(transformed_bands, sample_rate, target_freq).squeeze()
    transformed_bands = (transformed_bands - NORM_MEAN) / NORM_SD

    return transformed_bands


def stockwell_one_channel(data, fmin_samples, fmax_samples, df, band, gamma=15, window_type='kazemi'):
    # Returns [1, bands, time]
    data = data.ravel()
    n = len(data)
    extended_data = np.concatenate([data[::-1], data, data[::-1]])
    stock = st.st(extended_data, fmin_samples, fmax_samples, gamma, win_type=window_type)
    stock = stock[:, n:2 * n]
    stock = np.abs(stock)
    freqs = np.arange(stock.shape[0]) * df

    return average_into_bands(stock, freqs, band)


class DataCruncher:
    def __init__(self, n_processes=mp.cpu_count()):
        self.pool = mp.Pool(
            processes=n_processes,
            initializer=init_pool_worker,
        )

    def __call__(self, sample, sampling_rate):
        return preprocess_data(sample, sampling_rate, self.pool)

    def __del__(self):
        self.pool.close()
        self.pool.terminate()
        self.pool.join()


if __name__ == '__main__':
    preprocessing_object = DataCruncher()
    data = np.random.randn(1200, 10)
    import time

    start = time.time()
    for k in range(100):
        _ = preprocessing_object(data, 100)
    print((time.time() - start) / 100)

    a = preprocessing_object(data, 100)
    print(a.shape)
