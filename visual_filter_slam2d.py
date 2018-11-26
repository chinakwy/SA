import numpy as np
from visual_translated_y import translated_y
from scipy.linalg import block_diag


scale = 100


class VisualFilterSLAM2d:
    """ This block implements common visualization code for landmark-based
     Simultaneous Localization and Mapping (SLAM) algorithms. Use it as a
     "base class" for blocks that implement an acutal SLAM filter algorithm.
     Format of expected block inputs:
     - [platform] - irrelevant
     - [sensor] - irrelevant
     - landmarks - Nx2 array of landmark positions
     Expected output format: struct with fields
     - .featurePositions: Nx2 array of estimated landmark positions, one per row
     - .landmarkIds: feature-to-landmark associations, Nx1 column vector of indices into the landmark input
     - .featureCovariances: Nx3 array where each row [sigma_xx, sigma_xy, sigma_yy] describes the uncertainty of a
        landmark position estimation


     An optional parameter featureSigmaScale may be added (by the
     subclassing block or in the experiment definition) to specify the size of
     the uncertainty ellipses of the landmarks. If it is not given, the value
     of sigmaScale from filter_localization2d will be used.
     """
    def __init__(self, canvas, landmarks):
        self.canvas = canvas
        self.sigmaScanle = 3
        self.feature_positions_all = []

        self.landmarks = landmarks
        self.draw_feature_positions = {}
        for i in range(len(self.landmarks)):
            self.draw_feature_positions[i] = self.canvas.create_oval(0, 0, 0, 0, fill='blue', outline='blue', width=2)

        self.covariance_ellipse = {}
        for i in range(len(self.landmarks)):
            self.covariance_ellipse[i] = self.canvas.create_polygon(0, 0, 0, 0, fill='', outline='blue', width=1)

        self.lm_associations = {}
        for i in range(len(self.landmarks)*3):
            self.lm_associations[i] = canvas.create_line(0, 0, 0, 0, fill='blue')

    def draw_features(self, filter_out):
        feature_positions = np.dot(np.array(filter_out['featurePositions'])[-1], scale)
        for a in range(len(feature_positions)):
            self.canvas.coords(self.draw_feature_positions[a],
                               feature_positions[a][0], translated_y(feature_positions[a][1]),
                               feature_positions[a][0]+1, translated_y(feature_positions[a][1]+1))

        # for _ in range(len(feature_positions), len(self.landmarks)):
        #     self.canvas.coords(self.draw_feature_positions[_], 0, 0, 0, 0)

    def draw_covariances(self, filter_out):
        M = len(np.array(filter_out['featurePositions'])[-1])
        covariance_ellipse_t = np.linspace(0, 2*np.pi, 10)
        sigma_scanle = self.sigmaScanle
        CS = np.hstack((np.cos(covariance_ellipse_t).reshape((len(covariance_ellipse_t), 1)),
                         np.sin(covariance_ellipse_t).reshape((len(covariance_ellipse_t), 1))))
        for i in range(M):
            cov = np.array(filter_out['featureCovariances'])[-1][i]
            eigval, eigvec = np.linalg.eig(np.array([[cov[0], cov[1]],
                                                     [cov[1], cov[2]]]))
            xy1 = np.diag(np.dot(sigma_scanle, np.power(np.abs(eigval), 0.5)))
            xy = np.dot(np.dot(CS, xy1), eigvec.T)
            xydata = []
            for _ in range(len(covariance_ellipse_t)):
                xdata = (xy[_][0] + np.array(filter_out['featurePositions'])[-1][i][0]) * scale
                ydata = translated_y((xy[_][1] + np.array(filter_out['featurePositions'])[-1][i][1]) * scale)
                xydata.append(xdata)
                xydata.append(ydata)
            self.canvas.coords(self.covariance_ellipse[i], xydata)

    def draw_lm_associations(self, filter_out):
            # in case no reference landmarks are provided, we skip drawing the
            # feature<->landmark associations
        M = len(np.array(filter_out['landmarkIds'])[-1])
        coords = np.full((3 * M + 1, 2), np.inf)
        for i in range(M):
            lmId = int(np.array(filter_out['landmarkIds'])[-1][i])
            if lmId > 0 & lmId <= len(self.landmarks):
                coords[3*i, :] = self.landmarks[lmId, :] * scale
                coords[3*i+1, :] = np.array(filter_out['featurePositions'])[-1][i, :] * scale

        for i in range(3*M):
            self.canvas.coords(self.lm_associations[i], coords[i, 0], translated_y(coords[i, 1]),
                               coords[i+1, 0], translated_y(coords[i+1, 1]))

        rest_ids = list(set(np.arange(len(self.landmarks) * 3)).difference(set(np.arange(3 * M))))
        for _ in range(len(rest_ids)):
            self.canvas.coords(self.lm_associations[rest_ids[_]], 0, 0, 0, 0)
