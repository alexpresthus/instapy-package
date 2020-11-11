import cv2
import matplotlib.pyplot as plt
import numpy as np
import time as time

def python_color2sepia(img):
    """
    Takes an image formatted as a 3-dimensional array [H, W, C], where C is the color channel (RGB),
    applies sepia filter on each pixel's color channel element-wise (as unsigned ints), and returns the filtered version.
    Args:
        img: Numpy (unsigned) integer 3D array of the original image.
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

def sepia_image_python(input_filename, output_filename=None):
    """
    Returns a numpy (unsigned) integer 3D array of a sepia image of input_filename.
    Prints the filter function runtime, and displays a plot showing the before / after images side by side.
    If output_filename is supplied, the created image is also saved to the specified location.
    If not supplied, it is saved to a custom location (input_filename + "_sepia")
    Args:
        input_filename: The filename for the image to use.
        output_filename: The location for where to save the filtered image.
    Returns:
        <uint8> numpy.array: Numpy (unsigned) integer 3D array of the filtered image.
    """
    # Convert image to numpy array using cv2. Make a copy in order to display original image.
    image = cv2.cvtColor(cv2.imread(input_filename), cv2.COLOR_BGR2RGB)
    image_sepia = image.copy()

    # Call converter function and print runtime using time.time()
    t0 = time.time()
    image_sepia = python_color2sepia(image_sepia)
    t1 = time.time()
    print(f'Runtime python_color2sepia: {t1-t0} seconds')

    # Show difference in images side by side using matplotlib.pyplot
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.suptitle('Python for Instagram')
    ax1.imshow(image)
    ax1.set_title('Original image')
    ax2.imshow(image_sepia)
    ax2.set_title('Sepia filtered image')
    plt.show()

    # Save filtered image to new file
    if output_filename is None:
        output_filename = input_filename.split('.')[0] + '_sepia.' + input_filename.split('.')[1]

    status = cv2.imwrite(output_filename, image_sepia)
    print(f'Saving sepia image as "{output_filename}": {status}')

    return image_sepia
