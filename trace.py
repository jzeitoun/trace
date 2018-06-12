import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import cv2
import tifffile as tif

import transforms3d as t3d

class Cylinder(object):

    def __init__(self, radius=5, height=20):
        self.radius = radius
        self.height = height
        self.psi = 0
        self.theta = 0
        self.original_volume = self._makeGaussianCylinder(height=height, radius=radius)
        self.original_center = [height+radius, height+radius, radius]
        self.original_indices = np.argwhere(self.original_volume)
        self.original_values = self.original_volume[[*self.original_indices.T]]
        self.translated_volume = np.zeros(self.original_volume.shape)

    @property
    def translated_indices(self):
        return np.where(self.translated_volume > 0)

    @property
    def translated_values(self):
        return self.translated_volume[[*self.translated_indices]]

    def rotate(self, psi=0, theta=0):
        '''
        Rotates cylinder by specified angles first in psi and then in theta.
        Angles are in degrees.
        '''
        self.psi = psi
        self.theta = theta
        # Translate center of cylinder to origin
        indices = self.original_indices - self.original_center
        rot_mat = t3d.euler.euler2mat(np.deg2rad(psi), 0, np.deg2rad(theta), axes='sxyz')
        new_indices = np.rint(rot_mat.dot(indices.T).T).astype(np.int64)
        # Translate center of cylinder back
        new_indices = new_indices + self.original_center
        translated_gaussian_cylinder = self.translated_volume
        translated_gaussian_cylinder[:] = 0
        translated_gaussian_cylinder[[*new_indices.T]] = self.original_values
        self.translated_volume = translated_gaussian_cylinder

    def render_voxels(self, volume_choice='translated'):
        volume = {'original': self.original_volume, 'translated': self.translated_volume}[volume_choice]
        colors = np.empty(volume.shape, dtype=object)
        volume_mask = volume > 0
        colors[volume_mask] = 'red'
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.voxels(volume_mask, facecolors=colors, edgecolor='k')
        plt.show()

    def render_gauss(self, volume_choice='translated'):
        volume = {'original': self.original_volume, 'translated': self.translated_volume}[volume_choice]
        colors = np.zeros(volume.shape + (4,))
        colors[..., 0] = 1
        colors[..., 3] = volume
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.voxels(volume > 0, facecolors=colors)
        plt.show()

    def _makeGaussianCylinder(self, height, radius):
        '''
        Creates a gaussian cylinder.
        '''
        # Execution time with size of 51, height of 25, and fwhm of 5: 308us
        bool_cylinder = self._makeCylinderMask(height, radius)
        length, _, height = bool_cylinder.shape
        gaussian_kernel = self._makeGaussian(size=length, fwhm=radius)
        gaussian_cylinder = np.stack([gaussian_kernel] * height, axis=2)
        gaussian_cylinder[~bool_cylinder] = 0
        return gaussian_cylinder

    def _makeGaussian(self, size, fwhm = 3, center=None):
        '''
        Make a square gaussian kernel.

        size: the length of a side of the square
        fwhm: full-width-half-maximum, which can be thought of as an effective radius.
        '''
        x = np.arange(0, size, 1, float)
        y = x[:,np.newaxis]

        if center is None:
            x0 = y0 = size // 2
        else:
            x0 = center[0]
            y0 = center[1]

        return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

    def _makeCylinderMask(self, height, radius):
        '''
        Creates a cylindrical mask.
        '''
        # Execution time with default params: 1.46ms
        # Added radius padding to allow for edge of cylinder transforming beyond height.
        x,y,z = mgrid[
                -height-radius : height+radius+1,
                -height-radius : height+radius+1,
                -radius : height+radius
                ]
        return (x**2 + y**2 <= radius**2) & (z <= height) & (z >= 0)
