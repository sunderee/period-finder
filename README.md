# DOS Homework

Transcript of homework instructions from Slovenian:

> Given a 512x10 matrix Y representing 10 signals each 512 samples long, determine which on of them is periodic and compute
> its period. All signals, but one, have added noise. You need to solve the task wihout the use of DFT (Direct Fourier
> Transform).

Matrix is stored as a MatLab file in `assets` directory. I've used SciPy's `loadmat()` method to open it. Another minor
tweak is the transpose of original matrix for easier processing.

## Solution

Without Fourier transform we need to compute autocorrelation. Unfortunately, this is not a straightforward procedure if
we rely on NumPy. This is the method I've used to compute autocorrelation (it is an adaptation from
[this thread](https://stackoverflow.com/questions/47351483/autocorrelation-to-estimate-periodicity-with-numpy)):

```python
import numpy as np

def compute_autocorrelation(input_signal):
    size = input_signal.size
    norm = input_signal - np.mean(input_signal)
    correlation = np.correlate(norm, norm, mode='same')
    return correlation[size // 2 + 1:] / \
           (input_signal.var() * np.arange(size - 1, size // 2, -1))
```

The autocorrelations array is 255 elements long, and the next step is finding peaks and computing the period between
them.

With the help of SciPy's `find_peaks` function, we can do exactly that, but we have to provide some additional parameters.
I've tweaked them based on the autocorrelation plots. Next, we compute the difference between peaks and standard deviation:
if the distance between two consecutive peaks is not constant, the standard deviation will detect it.

**Solution: fourth (4th) sample is periodic with a period of 16 samples**.

## Usage

Preferred method when working with Python is to generate a virtual environment and install necessary dependencies there.
If you're on Linux or MacOS:

```bash
# Create a virtual environment
$ virtualenv venv

# Start the virtual environment
$ venv/bin/activate

# Install dependencies
$ pip install -r requirements.txt
```

IDEs such as PyCharm should generate the environment for you.
