# __author__: Gautam Sharma
# data:10/12/21
import numpy as np
import cv2 as cv2
from Preprocess import Preprocess

class LaneDetection:
    def __init__(self, **kwargs):
        self.preprocess = Preprocess(**kwargs)
        self.num_windows = 9  # Choose the number of sliding windows
        self.margin = 100  # Set the width of the windows +/- self.margin
        self.minpix = 50  # Set minimum number of pixels found to recenter window

    def findLanePixels(self,binary_warped):
        """

        :param binary_warped: binary warped image
        :return:
        leftx: x coordinates of left lane line
        lefty: y coordinates of left lane line
        rightx: x coordinates of right lane line
        righty: y coordinates of right lane line
        out_img: image with sliding windows drawn

        """
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        out_img_original = np.copy(out_img)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows - based on self.num_windows above and image shape
        window_height = np.int(binary_warped.shape[0]//self.num_windows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in self.num_windows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(self.num_windows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),
                          (win_xleft_high,win_y_high),(0,255,0), 2)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
                          (win_xright_high,win_y_high),(0,255,0), 2)

            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > self.minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, out_img, out_img_original
    def fit_polynomial(self,binary_warped):
        """

        :param binary_warped: binary warped image
        :return:
        left_fitx: x coordinates of the left lane after poly fit
        right_fitx: x coordinates of the right lane after poly fit
        ploty: list of 0 to y axis limit used to sample y values
        out_img_original: binary warped image
        """
        # Find our lane pixels first
        leftx, lefty, rightx, righty, out_img, out_img_original = self.findLanePixels(binary_warped)

        # Fit a second order polynomial to each using `np.polyfit`
        if len(leftx) > 1 and len(lefty) > 1:
            left_fit = np.polyfit(lefty, leftx, 2)
        else:
            print("No lane to fit")
            return
        if len(rightx) > 1 and len(righty) > 1:
            right_fit = np.polyfit(righty, rightx, 2)
        else:
            print("No lane to fit")
            return

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])

        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty

        return left_fitx, right_fitx, ploty, out_img_original


    def readVideoStream(self):
        cap = cv2.VideoCapture(self.preprocess.video_path)
        # Check if camera opened successfully
        if (cap.isOpened()== False):
            print("Error opening video stream or file")

        # Read until video is completed
        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            print(frame.shape)
            if ret == True:

                warped = self.preprocess.getPerspectiveTransform(frame)


                #line_edges1 = 255*self.preprocess.sobel(warped)
                line_edges2 = 255*self.preprocess.hueLightSaturation(warped)

                try:
                    left_fitx, right_fitx, ploty, out_img_original = ld.fit_polynomial(line_edges2)

                except:
                    print("No line detected! ")
                    continue

                for i in range(len(ploty)):
                    cv2.circle(warped, (int(left_fitx[i]), int(ploty[i])), 3,255)
                    cv2.circle(warped, (int(right_fitx[i]), int(ploty[i])), 3,255)
                cv2.transform(warped, self.preprocess.M)
                #input()
                reverse_perspective = self.preprocess.getReversePerspectiveTransform(warped)

                cv2.imshow("frame", reverse_perspective)

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
    ld = LaneDetection()
    ld.readVideoStream()

















