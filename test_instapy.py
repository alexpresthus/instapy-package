import numpy as np
import sys
import pytest
import cv2
import os

from instapy.all_implementations.numpy_color2gray import grayscale_image_numpy
from instapy.all_implementations.python_color2gray import grayscale_image_python
from instapy.all_implementations.numba_color2gray import grayscale_image_numba
from instapy.all_implementations.numpy_color2sepia import sepia_image_numpy
from instapy.all_implementations.python_color2sepia import sepia_image_python
from instapy.all_implementations.numba_color2sepia import sepia_image_numba

# Static global variables
sepia_matrix = np.array([[0.393, 0.769, 0.189],
                            [0.349, 0.686, 0.168],
                            [0.272, 0.534, 0.131]])

def setup_module(module):
    """
    py.test setup module
    Generates a 3D numpy array with pixel values randomly chosen from 0 to 255, and saves it to current working directory as "random.jpg"
    """
    print('\nSETUP')

    # Generate 3D numpy array with pixel values randomly chosen from 0 to 255
    image_random = np.random.randint(0, 255, size=(400, 600, 3))

    # Save random image
    status = cv2.imwrite("random.jpg", image_random)
    print(f'Saving random image as "random.jpg": {status}')

def teardown_module(module):
    """
    py.test teardown module
    Removes all files generated during test from current working directory
    """
    print('\nTEARDOWN')
    # Remove "random.jpg"
    file_paths = ['./random.jpg', './random_grayscale.jpg', './random_sepia.jpg']

    for file in file_paths:
        try:
            os.remove(file)
            print(f'Removed file {file}')
        except OSError as e:
            print(f'Error: {file}: {e.strerror}')

def test_grayscale_filter_functions():
    """
    Tests the grayscale_filter functions in all implementations (pure python, numpy and numba).
    """
    # Initialize test variables
    image_numpy = grayscale_image_numpy("random.jpg")
    image_python = grayscale_image_python("random.jpg")
    image_numba = grayscale_image_numba("random.jpg")
    h = np.random.randint(0, 255)
    w = np.random.randint(0, 255)
    expected_gray_val = int(np.dot(image_numpy[h,w],[.21,.72,.07]))
    expected_gray_pixel = [expected_gray_val, expected_gray_val, expected_gray_val]

    # Check that a random pixel value of the returned grayscale image is as expected
    # Numpy implementation
    assert set(image_numpy[h,w]) == pytest.approx(set(expected_gray_pixel), rel=1)
    # Python implementation
    assert set(image_python[h,w]) == pytest.approx(set(expected_gray_pixel), rel=1)
    # Numba implementation
    assert set(image_numba[h,w]) == pytest.approx(set(expected_gray_pixel), rel=1)

def test_sepia_filter_functions():
    """
    Tests the sepia_filter functions in all implementations (pure python, numpy and numba).
    Uses the random 3d array test-image "random.jpg" as argument for the filter functions of each implementation.
    Generates random values h, w between 0 and 255, to use as height and width index.
    Tests that this random index in every returned image is as expected.
    """
    # Initialize test variables
    image_numpy = sepia_image_numpy("random.jpg")
    image_python = sepia_image_python("random.jpg")
    image_numba = sepia_image_numba("random.jpg")
    h = np.random.randint(0, 255)
    w = np.random.randint(0, 255)
    expected_sepia_pixel = np.matmul(image_numpy[h,w],sepia_matrix.T)

    # Check that a random pixel value of the returned sepia image is as expected
    # Numpy implementation
    assert set(image_numpy[h,w]) == pytest.approx(set(expected_sepia_pixel), rel=1)
    # Python implementation
    assert set(image_python[h,w]) == pytest.approx(set(expected_sepia_pixel), rel=1)
    # Numba implementation
    assert set(image_numba[h,w]) == pytest.approx(set(expected_sepia_pixel), rel=1)
