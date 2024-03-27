import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d


def rgb2gray(img_color):
    # input:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image
    # return:
    #    img_gray - a h x w numpy ndarray (dtype = np.float64) holding
    #               the grayscale image
    img_gray = np.dot(img_color, [0.299, 0.587, 0.114])
    return img_gray


def smooth1D(img, sigma):
    # input :
    #    img - a h x w numpy ndarray holding the image to be smoothed
    #    sigma - sigma value of the 1D Gaussian function
    # return:
    #    img_smoothed - a h x w numpy ndarry holding the 1D smoothing result

    filter_size = int(sigma * (2 * np.log(1000)) ** 0.5)
    kernel = np.arange(-filter_size, filter_size + 1)
    gaussian_filter = np.exp((kernel**2) / -2 / (sigma**2))
    # gaussian_filter /= gaussian_filter.sum()
    img_filtered = convolve1d(img, gaussian_filter, 1, np.float64, "constant", 0, 0)
    weight_matrix = np.ones_like(img)
    weight_matrix_filtered = convolve1d(
        weight_matrix, gaussian_filter, 1, np.float64, "constant", 0, 0
    )
    img_smoothed = img_filtered / weight_matrix_filtered
    return img_smoothed


def smooth2D(img, sigma):
    # input:
    #    img - a h x w numpy ndarray holding the image to be smoothed
    #    sigma - sigma value of the Gaussian function
    # return:
    #    img_smoothed - a h x w numpy array holding the 2D smoothing result

    img_smoothed = smooth1D(img, sigma)
    img_smoothed = img_smoothed.T
    img_smoothed = smooth1D(img_smoothed, sigma)
    img_smoothed = img_smoothed.T

    return img_smoothed


def harris(img, sigma, threshold):
    # input:
    #    img - a h x w numpy ndarry holding the input image
    #    sigma - sigma value of the Gaussian function used in smoothing
    #    threshold - threshold value used for identifying corners
    # return:
    #    corners - a list of tuples (x, y, r) specifying the coordinates
    #              (up to sub-pixel accuracy) and cornerness value of each corner

    Ix = np.gradient(img, axis=0)
    Iy = np.gradient(img, axis=1)
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    IxIy = Ix * Iy
    Ix2_smoothed = smooth2D(Ix2, sigma)
    Iy2_smoothed = smooth2D(Iy2, sigma)
    IxIy_smoothed = smooth2D(IxIy, sigma)

    k = 0.04
    R = (Ix2_smoothed * Iy2_smoothed - IxIy_smoothed**2) - k * (
        Ix2_smoothed + Iy2_smoothed
    ) ** 2

    # perform quadratic approximation to local corners upto sub-pixel accuracy
    corners = []
    for i in range(1, R.shape[0] - 1):
        for j in range(1, R.shape[1] - 1):
            if (
                R[i, j] > R[i - 1, j - 1]
                and R[i, j] > R[i - 1, j]
                and R[i, j] > R[i - 1, j + 1]
                and R[i, j] > R[i, j - 1]
                and R[i, j] > R[i, j + 1]
                and R[i, j] > R[i + 1, j - 1]
                and R[i, j] > R[i + 1, j]
                and R[i, j] > R[i + 1, j + 1]
            ):

                f00 = R[i, j]
                f10 = R[i + 1, j]
                f01 = R[i, j + 1]
                f_10 = R[i - 1, j]
                f0_1 = R[i, j - 1]

                a = (f_10 + f10 - 2 * f00) / 2
                b = (f0_1 + f01 - 2 * f00) / 2
                c = (f10 - f_10) / 2
                d = (f01 - f0_1) / 2
                e = f00

                x = -c / (2 * a)
                y = -d / (2 * b)
                subpixel_x = i + x
                subpixel_y = j + y
                subpixel_R = a * x**2 + b * y**2 + c * x + d * y + e

                if subpixel_R >= threshold:
                    corners.append((subpixel_y, subpixel_x, subpixel_R))

    corners = [corner for corner in corners if corner[2] >= threshold]
    return sorted(corners, key=lambda corner: corner[2], reverse=True)


def show_corners(img_color, corners):
    # input:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image
    #    corners - a n x 3 numpy ndarray holding the coordinates and strengths
    #              of the detected corners (n being the number of corners)

    x = [corner[0] for corner in corners]
    y = [corner[1] for corner in corners]

    plt.ion()
    fig = plt.figure("Harris corner detection")
    plt.imshow(img_color)
    plt.plot(x, y, "r+", markersize=5)
    plt.show()
    plt.ginput(n=1, timeout=-1)
    plt.close(fig)


def load_image(inputfile):
    # input:
    #    inputfile - path of the image file
    # return:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image

    try:
        img_color = plt.imread(inputfile)
        return img_color
    except:
        print("Cannot open '{}'.".format(inputfile))
        sys.exit(1)


def save_corners(outputfile, corners):
    # input:
    #    outputfile - path of the output file
    #    corners - a n x 3 numpy ndarray holding the coordinates and strengths
    #              of the detected corners (n being the number of corners)

    try:
        file = open(outputfile, "w")
        file.write("{}\n".format(len(corners)))
        for corner in corners:
            file.write("{:.6e} {:.6e} {:.6e}\n".format(corner[0], corner[1], corner[2]))
        file.close()
    except:
        print("Error occurs in writing output to '{}'.".format(outputfile))
        sys.exit(1)


def load_corners(inputfile):
    # input:
    #    inputfile - path of the file containing corner detection output
    # return:
    #    corners - a n x 3 numpy ndarray holding the coordinates and strengths
    #              of the detected corners (n being the number of corners)

    try:
        file = open(inputfile, "r")
        line = file.readline()
        nc = int(line.strip())
        print("loading {} corners".format(nc))
        corners = np.zeros([nc, 3], dtype=np.float64)
        for i in range(nc):
            line = file.readline()
            x, y, r = line.split()
            corners[i] = [np.float64(x), np.float64(y), np.float64(r)]
        file.close()
        return corners
    except:
        print("Error occurs in loading corners from '{}'.".format(inputfile))
        sys.exit(1)


################################################################################
#  main
################################################################################
def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-i", "--image", type=str, default="grid1.jpg", help="filename of input image"
    )
    parser.add_argument(
        "-s",
        "--sigma",
        type=float,
        default=1.0,
        help="sigma value for Gaussain filter (default = 1.0)",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=1e6,
        help="threshold value for corner detection (default = 1e6)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="filename for outputting corner detection result",
    )
    args = parser.parse_args()

    print("------------------------------")
    print("input file : {}".format(args.image))
    print("sigma      : {:.2f}".format(args.sigma))
    print("threshold  : {:.2e}".format(args.threshold))
    print("output file: {}".format(args.output))
    print("------------------------------")

    # load the image
    img_color = load_image(args.image)
    print("'{}' loaded...".format(args.image))

    # plt.imshow(np.uint8(img_color))
    # plt.show()

    # perform RGB to gray conversion
    print("perform RGB to grayscale conversion...")
    img_gray = rgb2gray(img_color)

    # plt.imshow(np.float32(img_gray), cmap = 'gray')
    # plt.show()

    # img_gaussian = smooth2D(rgb2gray(img_color),args.sigma)
    # plt.imshow(np.float32(img_gaussian), cmap = 'gray')
    # plt.show()

    # perform corner detection
    print("perform Harris corner detection...")
    corners = harris(img_gray, args.sigma, args.threshold)

    # plot the corners
    print("{} corners detected...".format(len(corners)))
    show_corners(img_color, corners)

    if args.output:
        save_corners(args.output, corners)
        print("corners saved to '{}'...".format(args.output))


if __name__ == "__main__":
    main()
