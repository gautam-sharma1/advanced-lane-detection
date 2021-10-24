# __author__: Gautam Sharma
# data:10/12/21
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv2
from numpy.linalg import inv
class Preprocess:
    def __init__(self, hough_rho=1, hough_theta = np.pi/180, hough_threshold=13, hough_min_length=50, hough_max_line_gap = 10,\
            canny_low_thresh = 78, canny_high_thresh = 114, canny_kernel_size = 11, sobel_kernel_size = 9, sobel_thresh_min = 55,\
                 sobel_thresh_max = 100):
        """

        :param hough_rho: distance resolution in pixels of the Hough grid
        :param hough_theta: angular resolution in radians of the Hough grid
        :param hough_threshold: minimum number of votes (intersections in Hough grid cell)
        :param hough_min_length: minimum number of pixels making up a line
        :param hough_max_line_gap: maximum gap in pixels between connectable line segments
        :param canny_low_thresh: If a pixel gradient value is below the lower threshold, then it is rejected
        :param canny_high_thresh: If a pixel gradient is higher than the upper threshold, the pixel is accepted as an edge\
        If the pixel gradient is between the two thresholds, then it will be accepted only if it is connected to a pixel that is above the upper threshold.
        Canny recommended a upper:lower ratio between 2:1 and 3:1.
        :param canny_kernel_size: gaussian blur parameter
        :param sobel_kernel_size = kernel size used by sobel filter
        :param sobel_thresh_min = minimum value for the pixel to be accepted as a lane pixel
        :param sobel_thresh_max = maximum value for the pixel to be accepted as a lane pixel
        """
        self.rho = hough_rho
        self.theta = hough_theta
        self.hough_threshold = hough_threshold
        self.hough_min_length = hough_min_length
        self.hough_max_line_gap = hough_max_line_gap
        self.canny_low_thresh = canny_low_thresh
        self.canny_high_thresh = canny_high_thresh
        self.canny_kernel_size = canny_kernel_size
        self.sobel_kernel_size = sobel_kernel_size
        self.sobel_thresh_min = sobel_thresh_min
        self.sobel_thresh_max = sobel_thresh_max
        self.video_path = '../data/video_1.mp4'
        self.edges = None
        self.masked_edges = None
        self.src = np.float32(
            [[350,550],
             [600,550],
            [750,650],
             [350, 650]]
        )

        self.dst = np.float32(
            [[200,0],
             [800,0],
             [800,700],
             [200,700]]
        )

        self.dst2 = np.float32(
            [[0,0],
             [1080,0],
             [1080,720],
             [0,720]]
        )

    def __repr__(self):
        return repr('Preprocessing Class')

    def __str__(self):
        # too lazy to pass in all the parameters :P
        return 'a Preprocessing class having rho =  {self.rho}'.format(self=self)


    def getCannyEdges(self, gray_image):
        """

        :param gray_image: gray scale image
        :return: canny edges
        """
        if gray_image is None:
            TypeError("Image is None!")

        # TODO : assert image size
        # Define a kernel size and apply Gaussian smoothing
        self.blur_gray = cv2.GaussianBlur(gray_image,(self.canny_kernel_size, self.canny_kernel_size),0)
        self.edges = cv2.Canny(self.blur_gray, self.canny_low_thresh, self.canny_high_thresh)
        return self.edges

    def maskEdges(self,ignore_mask_color):
        """
        :param ignore_mask_color: color for cv2.fillPoly
        :return:
        """

        if self.edges is None:
            ValueError("Canny Edges are not defined. Run Canny Detector first!")
            exit()
        # This time we are defining a four sided polygon to mask

        # image of all pixels set to 0
        self.mask = np.zeros_like(self.edges)
        self.y, self.x = self.mask.shape
        #TODO: remove hard coded values
        vertices = np.array([[(400, 440),(220,self.y-50),(800, self.y-50),(800,440) ]], dtype=np.int32)
        cv2.fillPoly(self.mask, vertices, ignore_mask_color)
        self.masked_edges = cv2.bitwise_and(self.edges, self.mask)
        return self.masked_edges

    def getPerspectiveTransform(self, image):
        """

        :param image: colored image that needs to be transformed
        :return: warped image
        """
        img_size = (image.shape[1], image.shape[0])

        # TODO: change this depending upon camera output

        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        warped = cv2.warpPerspective(image, self.M, img_size, flags=cv2.INTER_CUBIC)
        return warped

    def getReversePerspectiveTransform(self,image):
        img_size = (image.shape[1], image.shape[0])
        #M = cv2.getPerspectiveTransform(self.dst, self.src)
        M_inv = inv(self.M)
        return cv2.warpPerspective(image, M_inv, img_size, flags=cv2.INTER_CUBIC)

    def sobel(self, image):
        """

        :param image: 3D image
        :return: binary image having brigh pixels for lane markings and dark pixels elsewhere
        """
        assert image is not None
        gray = Preprocess.cvt2Gray(image)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0,self.sobel_kernel_size)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1,self.sobel_kernel_size)
        abs_sobel = np.sqrt(sobelx**2 + sobely**2)
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        binary = np.zeros_like(scaled_sobel)

        # mask the lane region by white pixels
        binary[(scaled_sobel >= self.sobel_thresh_min) & (scaled_sobel <= self.sobel_thresh_max)] = 1
        # import matplotlib.pyplot as plt
        # plt.imshow(binary,cmap="gray")
        # plt.show()
        # input()
        # mask the region that is not lanes by 0 i.e. black pixels
        # TODO: change this depending upon camera output
        binary[:, :200] = 0
        binary[:, 870:] = 0
        #binary[:,450:750] = 0
        binary[:200,:] = 0
        # binary[670:, :] = 0

        return binary

    def hueLightSaturation(self, image):
        """
        Source: https://en.wikipedia.org/wiki/HSL_and_HSV
        HSL (for hue, saturation, lightness) and HSV (for hue, saturation, value; also known as HSB, for hue, saturation
        , brightness) are alternative representations of the RGB color model, designed in the 1970s by computer graphics
        researchers to more closely align with the way human vision perceives color-making attributes. In these models,
        colors of each hue are arranged in a radial slice, around a central axis of neutral colors which ranges from
        black at the bottom to white at the top.
        The HSL representation models the way different paints mix together to create colour in the real world, with the
        lightness dimension resembling the varying amounts of black or white paint in the mixture (e.g. to create
        "light red", a red pigment can be mixed with white paint; this white paint corresponds to a high "lightness"
        value in the HSL representation). Fully saturated colors are placed around a circle at a lightness value of ½,
        with a lightness value of 0 or 1 corresponding to fully black or white, respectively.

        :param image: 3D image
        :return: 1D image after being preprocessed by custom HLS combinations that have proven to work well with the
        input data
        """

        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

        R = image[:,:,2]
        H = hls[:,:,0]
        L = hls[:,:,1]
        S = hls[:,:,2]

        # threshold values that have been experimented with and work well
        thresh_S = (160,225)
        thresh_L = (190, 255)
        thresh_R = (170,255)

        binary_S = np.zeros_like(S)
        binary_L = np.zeros_like(L)
        binary_R = np.zeros_like(R)

        binary_S[(S > thresh_S[0]) & (S <= thresh_S[1])] = 1
        binary_L[(L > thresh_L[0]) & (L <= thresh_L[1])] = 1
        binary_R[(R > thresh_R[0]) & (R <= thresh_R[1])] = 1

        net_img = binary_R + binary_L + binary_S
        net_img[:, :200] = 0
        net_img[:, 870:] = 0
        #binary[:,450:750] = 0
        net_img[:200,:] = 0
        # net_img[670:, :] = 0
        return net_img

    @staticmethod
    def cvt2Gray(image, format = "BGR"):
        """

        :param image: cv2 image in BGR format
        :return: cv2 gray image
        """
        if format is "BGR":
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif format is "RGB":
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        else:
            ValueError("Incorrect value assigned to format!")

    def _readVideoStream(self):
        cap = cv2.VideoCapture(self.video_path)
        # Check if camera opened successfully
        if (cap.isOpened()== False):
            print("Error opening video stream or file")

        # Read until video is completed
        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            print(frame.shape)
            if ret == True:
                warped = self.getPerspectiveTransform(frame)


                line_edges1 = 255*self.sobel(warped)
                line_edges2 = 255*self.hueLightSaturation(line_edges1)

                #input()

                cv2.imshow("frame", line_edges2)

                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            # Break the loop
            else:
                break

        # When everything done, release the video capture object
        cap.release()

        # Closes all the frames
        cv2.destroyAllWindows()

if __name__ == "__main__":
    p = Preprocess()
    p._readVideoStream()