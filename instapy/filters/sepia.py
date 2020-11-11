# sepia.py
import cv2
import numpy as np
from numba import jit
import matplotlib.pyplot as plt

class Sepia:
    def __init__(self):
        """
        """
        pass

    def sepia_image(self, input_filename, implementation="numpy", effect=1, output_filename=None):
        """
        Returns a numpy (unsigned) integer 3D array of a sepia image of input_filename.
        Prints the filter function runtime, and displays a plot showing the before / after images side by side.
        If effect is supplied, sepia effect (0-100%) is applied according to effect (0-1). If not, it applies full (100%) effect.
        If output_filename is supplied, the created image is also saved to the specified location. If not supplied, it is saved to a custom location (input_filename + "_sepia")
        Args:
            input_filename: The filename for the image to use.
            effect: Value from 0 - 1 to define 0-100% Sepia effect. Defaults to 1.
            output_filename: The location for where to save the filtered image. Defaults to None.
        Returns:
            <uint8> numpy.array: Numpy (unsigned) integer 3D array of the filtered image.
        """
        # Convert image to numpy array using cv2. Make a copy in order to display original image.
        image = cv2.cvtColor(cv2.imread(input_filename), cv2.COLOR_BGR2RGB)
        image_sepia = image.copy()

        # Call converter function
        if implementation == "numpy":
            image_sepia = numpy_color2sepia(image_sepia, effect)
        elif implementation == "python":
            image_sepia = python_color2sepia(image_sepia, effect)
        else:
            image_sepia = numba_color2sepua(image_sepia, effect)

        # Save filtered image to new file
        if output_filename is not None:
            status = cv2.imwrite(output_filename, image_sepia)
            print(f'Saving sepia image as "{output_filename}": {status}')

        self.image_original = image
        self.image_sepia = image_sepia

        return image_sepia

    def display_images(self):
        # Show difference in images side by side using matplotlib.pyplot
        fig, (ax1, ax2) = plt.subplots(1,2)
        fig.suptitle('Python for Instagram')
        ax1.imshow(self.image_original)
        ax1.set_title('Original image')
        ax2.imshow(self.image_sepia)
        ax2.set_title('Sepia filtered image')
        plt.show()


def numpy_color2sepia(img, effect):
    """
    Takes an image formatted as a 3-dimensional array [H, W, C], where C is the color channel (RGB),
    applies sepia filter on the image array and overwrites the color channel array (C), and returns the filtered version.
    If effect is supplied, sepia effect (0-100%) is applied according to effect (0-1). If not, it applies full (100%) effect.
    Args:
        img: Numpy (unsigned) integer 3D array of the original image.
        effect: Value from 0 - 1 to define 0-100% Sepia effect.
    Returns:
        <uint8> numpy.array: Numpy (unsigned) integer 3D array of the filtered image.
    """
    k = 1
    if (effect <= 1 and effect >= 0):
        k = effect

    sepia_matrix_effect = np.array([[ 0.393 + 0.607 * (1 - k), 0.769 - 0.769 * (1 - k), 0.189 - 0.189 * (1 - k)],
                                [ 0.349 - 0.349 * (1 - k), 0.686 + 0.314 * (1 - k), 0.168 - 0.168 * (1 - k)],
                                [ 0.272 - 0.272 * (1 - k), 0.534 - 0.534 * (1 - k), 0.131 + 0.869 * (1 - k)]])
    img2 = np.array([np.matmul(x, sepia_matrix_effect.T) for x in img])
    img2[np.where(img2>255)] = 255

    return img2.astype("uint8")

def python_color2sepia(img, effect):
    """
    Takes an image formatted as a 3-dimensional array [H, W, C], where C is the color channel (RGB),
    applies sepia filter on each pixel's color channel element-wise (as unsigned ints), and returns the filtered version.
    If effect is supplied, sepia effect (0-100%) is applied according to effect (0-1). If not, it applies full (100%) effect.
    Args:
        img: Numpy (unsigned) integer 3D array of the original image.
        effect: Value from 0 - 1 to define 0-100% Sepia effect.
    Returns:
        <uint8> numpy.array: Numpy (unsigned) integer 3D array of the filtered image.
    """
    H = np.shape(img)[0]
    W = np.shape(img)[1]
    C = np.shape(img)[2]

    k = 1
    if (effect <= 1 and effect >= 0):
        k = effect

    for row in range(H):
        for col in range(W):
            r, g, b = img[row, col]

            rfilter = int(r * (0.393 + 0.607 * (1 - k)) + g * (0.769 - 0.769 * (1 - k)) + b * (0.189 - 0.189 * (1 - k)))
            gfilter = int(r * (0.349 - 0.349 * (1 - k)) + g * ( 0.686 + 0.314 * (1 - k)) + b * (0.168 - 0.168 * (1 - k)))
            bfilter = int(r * (0.272 - 0.272 * (1 - k)) + g * (0.534 - 0.534 * (1 - k)) + b * (0.131 + 0.869 * (1 - k)))

            if rfilter > 255:
                rfilter = 255

            if gfilter > 255:
                gfilter = 255

            if bfilter > 255:
                bfilter = 255

            img[row, col] = (rfilter, gfilter, bfilter)

    return img

@jit
def numba_color2sepia(img, effect):
    """
    Takes an image formatted as a 3-dimensional array [H, W, C], where C is the color channel (RGB),
    applies sepia filter on each pixel's color channel element-wise (as unsigned ints), and returns the filtered version.
    If effect is supplied, sepia effect (0-100%) is applied according to effect (0-1). If not, it applies full (100%) effect.
    Args:
        img: Numpy (unsigned) integer 3D array of the original image.
        effect: Value from 0 - 1 to define 0-100% Sepia effect.
    Returns:
        <uint8> numpy.array: Numpy (unsigned) integer 3D array of the filtered image.
    """
    H = np.shape(img)[0]
    W = np.shape(img)[1]
    C = np.shape(img)[2]

    k = 1
    if (effect <= 1 and effect >= 0):
        k = effect

    for row in range(H):
        for col in range(W):
            r, g, b = img[row, col]

            rfilter = int(r * (0.393 + 0.607 * (1 - k)) + g * (0.769 - 0.769 * (1 - k)) + b * (0.189 - 0.189 * (1 - k)))
            gfilter = int(r * (0.349 - 0.349 * (1 - k)) + g * ( 0.686 + 0.314 * (1 - k)) + b * (0.168 - 0.168 * (1 - k)))
            bfilter = int(r * (0.272 - 0.272 * (1 - k)) + g * (0.534 - 0.534 * (1 - k)) + b * (0.131 + 0.869 * (1 - k)))

            if rfilter > 255:
                rfilter = 255

            if gfilter > 255:
                gfilter = 255

            if bfilter > 255:
                bfilter = 255

            img[row, col] = (rfilter, gfilter, bfilter)

    return img
