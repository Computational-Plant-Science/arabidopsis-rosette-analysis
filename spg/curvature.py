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
