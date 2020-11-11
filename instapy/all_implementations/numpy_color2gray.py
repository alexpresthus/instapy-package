import cv2
import matplotlib.pyplot as plt
import numpy as np
import time as time

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

def grayscale_image_numpy(input_filename, output_filename=None):
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
    image_gray = numpy_color2gray(image_gray)
    t1 = time.time()
    print(f'Runtime numpy_color2gray: {t1-t0} seconds')

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
