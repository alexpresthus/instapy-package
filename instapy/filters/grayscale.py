# grayscale.py
import cv2
import numpy as np
from numba import jit

class Grayscale:
    def __init__(self):
        """
        """
        pass

    def grayscale_image(self, input_filename, implementation="numpy", output_filename=None):
        """
        Returns a numpy (unsigned) integer 3D array of a gray image of input_filename.
        If output_filename is supplied, the created image is also saved to the specified location.
        Args:
            input_filename: The filename for the image to use.
            output_filename: The location for where to save the filtered image.
        Returns:
            <uint8> numpy.array: Numpy (unsigned) integer 3D array of the filtered image.
        """
        # Convert image to numpy array using cv2.
        image = cv2.cvtColor(cv2.imread(input_filename), cv2.COLOR_BGR2RGB)
        image_gray = image.copy()

        # Call converter function
        if implementation == "numpy":
            image_gray = numpy_color2gray(image_gray)
        elif implementation == "python":
            image_gray = python_color2gray(image_gray)
        else:
            image_gray = numba_color2gray(image_gray)

        # Save filtered image to new file
        if output_filename is not None:
            status = cv2.imwrite(output_filename, image_gray)
            print(f'Saving grayscaled image as "{output_filename}": {status}')

        self.image_original = image
        self.image_gray = image_gray

        return image_gray

    def display_images(self):
        # Show difference in images side by side using matplotlib.pyplot
        fig, (ax1, ax2) = plt.subplots(1,2)
        fig.suptitle('Python for Instagram')
        ax1.imshow(self.image_original)
        ax1.set_title('Original image')
        ax2.imshow(self.image_gray)
        ax2.set_title('Grayscale image')
        plt.show()

def numpy_color2gray(img):
    """
    Takes an image formatted as a 3-dimensional array [H, W, C], where C is the color channel (RGB),
    applies grayscale filter on the image array by multiplying each picture RGB value by grayscale weights, then returns the filtered version.
    Args:
        img: Numpy (unsigned) integer 3D array of the original image.
    Returns:
        <uint8> numpy.array: Numpy (unsigned) integer 3D array of the filtered image.
    """
    gray_array = np.array([[0.21, 0.72, 0.07],
                            [0.21, 0.72, 0.07],
                            [0.21, 0.72, 0.07]])
    img2 = np.array([np.matmul(x, gray_array.T) for x in img])
    img2[np.where(img2>255)] = 255
    return img2.astype("uint8")

def python_color2gray(img):
    """
    Takes an image formatted as a 3-dimensional array [H, W, C], where C is the color channel (RGB),
    applies grayscale filter on each pixel's color channel element-wise (as unsigned ints), and returns the filtered version.
    Args:
        img: Numpy (unsigned) integer 3D array of the original image.
    Returns:
        <uint8> numpy.array: Numpy (unsigned) integer 3D array of the filtered image.
    """
    H = np.shape(img)[0]
    W = np.shape(img)[1]
    C = np.shape(img)[2]

    for row in range(H):
        for col in range(W):
            r, g, b = img[row, col]
            gray = int(r * 0.21 + g * 0.72 + b * 0.07)
            if gray > 255:
                gray = 255
            img[row, col] = (gray, gray, gray)

    return img

@jit
def numba_color2gray(img):
    """
    Takes an image formatted as a 3-dimensional array [H, W, C], where C is the color channel (RGB),
    applies grayscale filter on each pixel's color channel element-wise (as unsigned ints), and returns the filtered version.
    Args:
        img: Numpy (unsigned) integer 3D array of the original image.
    Returns:
        <uint8> numpy.array: Numpy (unsigned) integer 3D array of the filtered image.
    """
    H = np.shape(img)[0]
    W = np.shape(img)[1]
    C = np.shape(img)[2]

    for row in range(H):
        for col in range(W):
            r, g, b = img[row, col]
            gray = int(r * 0.21 + g * 0.72 + b * 0.07)
            if gray > 255:
                gray = 255
            img[row, col] = (gray, gray, gray)

    return img
