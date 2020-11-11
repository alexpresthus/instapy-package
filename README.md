## Python Instapy
A Python package - Instapy

Contains modules with functions to apply filters to images.

The package contains different modules for different implementations; numpy, numba and plain Python.

### Required dependencies and packages

Dependencies
* python 3.8
* pip 20.1.1

Python packages
* numpy 1.18.5
* pytest 5.4.3

Install required packages using:
```
pip install numpy, pytest
```

### instapy

#### Package info

Subpackage "filters" contains modules and functions implemented for grayscale filter and sepia filter, utilizing numpy for image array arithmetics.

#### Install
Install instapy using:
```
pip install .
```
from directory where setup.py is located.

### How to run scripts

After installing instapy (see instapy installation section), the instapy package is accessible system wide.

To access the package from within another python script, use:

```
import instapy
```

To access the package from shell, use:

```
instapy [-h] -f FILE (-se | -g) [-ef EFFECT] [-sc SCALE] [-i {numpy,numba,python}] [-d] [-o OUT]
```

### How to run tests

Run pytest using:

```
pytest
```
from directory where test_instapy.py is located.

### Additional comments

Grayscale functions:
* matplotlib.pyplt seems to have trouble displaying images of shape (H, W), without a channel-array. In order to display the images properly, I let the shape of gray image arrays be (H, W, C), with C = 3, filling all channels with the grayscale value, instead of having gray image arrays be just shaped (H, W).

pytest.ini
* Initialization file for pytest. Sets flag -p no:warnings, to bypass pytest internal deprecation warning (when using custom packages)
