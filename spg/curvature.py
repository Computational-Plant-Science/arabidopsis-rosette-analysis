import traceback

import cv2
import numpy as np
from scipy import optimize


class ComputeCurvature:

    def __init__(self, x, y):
        self.xc = 0  # X-coordinate of circle center
        self.yc = 0  # Y-coordinate of circle center
        self.r = 0  # Radius of the circle
        self.xx = np.array([])  # Data points
        self.yy = np.array([])  # Data points
        self.x = x  # X-coordinate of circle center
        self.y = y  # Y-coordinate of circle center

    def calc_r(self, xc, yc):
        """
        Calculate the distance of each 2D points from the center (xc, yc)
        """

        return np.sqrt((self.xx - xc) ** 2 + (self.yy - yc) ** 2)

    def f(self, c):
        """
        Calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc)
        """

        ri = self.calc_r(*c)
        return ri - ri.mean()

    def df(self, c):
        """
        Jacobian of f_2b
        The axis corresponding to derivatives must be coherent with the col_deriv option of leastsq
        """

        xc, yc = c
        df_dc = np.empty((len(c), self.x.size))

        ri = self.calc_r(xc, yc)
        df_dc[0] = (xc - self.x) / ri  # dR/dxc
        df_dc[1] = (yc - self.y) / ri  # dR/dyc
        df_dc = df_dc - df_dc.mean(axis=1)[:, np.newaxis]
        return df_dc

    def fit(self, xx, yy):
        self.xx = xx
        self.yy = yy
        center_estimate = np.r_[np.mean(xx), np.mean(yy)]
        center = optimize.leastsq(self.f, center_estimate, Dfun=self.df, col_deriv=True)[0]

        self.xc, self.yc = center
        ri = self.calc_r(*center)
        self.r = ri.mean()

        return 1 / self.r  # return the curvature


def compute_curv(orig, labels):
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

    curv_sum = 0.0
    count = 0
    # curvature computation
    # loop over the unique labels returned by the Watershed algorithm
    for index, label in enumerate(np.unique(labels), start=1):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue

        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255

        # detect contours in the mask and grab the largest one
        # cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        c = max(contours, key=cv2.contourArea)

        # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        label_trait = cv2.circle(orig, (int(x), int(y)), 3, (0, 255, 0), 2)
        label_trait = cv2.putText(orig, "#{}".format(label), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # cv2.putText(orig, "#{}".format(curvature), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if len(c) >= 5:
            try:
                label_trait = cv2.drawContours(orig, [c], -1, (255, 0, 0), 2)
                ellipse = cv2.fitEllipse(c)
                label_trait = cv2.ellipse(orig, ellipse, (0, 255, 0), 2)

                c_np = np.vstack(c).squeeze()
                count += 1

                x = c_np[:, 0]
                y = c_np[:, 1]

                comp_curv = ComputeCurvature(x, y)
                curvature = comp_curv.fit(x, y)

                curv_sum = curv_sum + curvature
            except:
                print(traceback.format_exc())
        else:
            # optional to "delete" the small contours
            label_trait = cv2.drawContours(orig, [c], -1, (0, 0, 255), 2)
            print("lack of enough points to fit ellipse")

    if count > 0:
        print('Average curvature: {0:.2f}'.format(curv_sum / count))
    else:
        count = 1.0

    return curv_sum / count, label_trait