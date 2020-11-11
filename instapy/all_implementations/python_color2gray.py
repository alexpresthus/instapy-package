import cv2
import matplotlib.pyplot as plt
import numpy as np
import time as time

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

def grayscale_image_python(input_filename, output_filename=None):
    """
    Returns a numpy (unsigned) integer 3D array of a gray image of input_filename.
    Prints the filter function runtime, and displays a plot showing the before / after images side by side.
    If output_filename is supplied, the created image is also saved to the specified location.
    If not supplied, it is saved to a custom location (input_filename + "_grayscale")
    Args:
        input_filename: The filename for the image to use.
        output_filename: The location for where to save the filtered image.
    Returns:
        <uint8> numpy.array: Numpy (unsigned) integer 3D array of the filtered image.
    """
    # Convert image to numpy array using cv2. Make a copy in order to display original image.
    image = cv2.cvtColor(cv2.imread(input_filename), cv2.COLOR_BGR2RGB)
    image_gray = image.copy()

    # Call converter function and print runtime using time.time()
    t0 = time.time()
    image_gray = python_color2gray(image_gray)
    t1 = time.time()
    print(f'Runtime python_color2gray: {t1-t0} seconds')

    # Show difference in images side by side using matplotlib.pyplot
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.suptitle('Python for Instagram')
    ax1.imshow(image)
    ax1.set_title('Original image')
    ax2.imshow(image_gray)
    ax2.set_title('Grayscale image')
    #plt.show()

    # Save filtered image to new file
    if output_filename is None:
        output_filename = input_filename.split('.')[0] + '_grayscale.' + input_filename.split('.')[1]

    status = cv2.imwrite(output_filename, image_gray)
    print(f'Saving grayscaled image as "{output_filename}": {status}')

    return image_gray
