import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy.ndimage.interpolation import map_coordinates, zoom

import transforms3d as t3d

class Cylinder(object):

    def __init__(self, radius=4, height=10, psi=0, theta=0, psf=1.5, first=False):
        self.radius = radius
        self.height = height
        self.psf = psf
        self.first = first
        self.psi = psi # angle around z
        self.theta = theta # angle around x
        self.reference_point = [height+radius, height+radius, height+radius]
        self.original_volume = np.array(self._make_gaussian_cylinder(radius=radius, height=height))
        self.original_coords = np.argwhere(self.original_volume) # returns coordinates where cylinder exists. An N x 3 array.
        self.translated_volume = self.original_volume.copy() # initialization for rotation

########################################################################################################################
    @property
    def translated_coords(self):
        return np.argwhere(self.translated_volume > 0)

    @property
    def scaled_coords(self):
        return np.argwhere(self.scaled_volume > 0)

    @property
    def translated_values(self):
        return self.translated_volume[[*self.translated_coords.T]]

    @property
    def scaled_values(self):
        return self.scaled_volume[[*self.scaled_coords.T]]

########################################################################################################################
# ** BASIC CYLINDER FUNCTIONS: CREATING AND ROTATING ** #
########################################################################################################################

    def _make_cylinder_mask(self, radius, height):
        """Returns a cylinder in a 3d array of Booleans. Array values of 'True' make up the volume of the cylinder.
          Execution time with default params: 1.46ms
          Added radius padding to allow for edge of cylinder transforming beyond height.
        """
        if self.first:
            x,y,z = np.array(np.mgrid[-height - radius:height + radius + 1, -height - radius:height + radius + 1, -height - radius:height + radius + 1])
        else:
            x,y,z = np.array(np.mgrid[-height - radius:height + radius + 1, -height - radius:height + radius + 1, -radius:height + radius + 1])
        self.xyz_coords = np.array((x, y, z))
        return (x ** 2 + y ** 2 <= radius ** 2) & (z <= height - 1) & (z >= 0) # -1 from height b/c zero indexing

########################################################################################################################
    def _make_gaussian(self, diameter, fwhm=2, center=None):
        """Returns a 2D gaussian kernel. will be repeated along the height of a cylinder.
        Inputs:
            diameter: the length of a side of the square
            fwhm: full-width-half-maximum, which can be thought of as an effective radius.
        """
        x = np.arange(0, diameter, 1, dtype=float)
        y = x[:, np.newaxis]

        if center is None:
            x0 = y0 = diameter // 2
        # never used
        else:
            x0 = center[0]
            y0 = center[1]

        return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)

########################################################################################################################

    def _make_gaussian_cylinder(self, radius, height, center=None):
        """Returns a cylinder with a gaussian cross section.
          Cylinder is a 3d array 8bit values.
          Execution time with diameter of 51, height of 25, and fwhm of 5: 308us
        """
        # generate cylinder
        cylinder_mask = self._make_cylinder_mask(radius, height)
        mask_length, mask_width, mask_height = cylinder_mask.shape
        # generate gaussian profile cylinder
        gaussian_kernel = self._make_gaussian(mask_length, radius, center)
        gaussian_cylinder = np.stack([gaussian_kernel] * mask_height, axis=2)
        # mask out values outside cylinder mask by setting them to 0
        gaussian_cylinder[~cylinder_mask] = 0
        return gaussian_cylinder

########################################################################################################################

    def rotate(self, psi=0, theta=0):
        '''
        Rotates cylinder by specified angles first in psi and then by theta.
        Angles are in degrees.
        '''
        self.psi = psi
        self.theta = theta
        rot_mat = t3d.euler.euler2mat(np.deg2rad(psi), 0, np.deg2rad(theta), axes='rxyz')
        transformed_xyz_coords = rot_mat.dot(self.xyz_coords.reshape(3,-1)).reshape(self.xyz_coords.shape)
        corrected_transformed_xyz_coords = (transformed_xyz_coords.reshape(3,-1).T + self.reference_point).T.reshape(self.xyz_coords.shape)
        map_coordinates(self.original_volume, corrected_transformed_xyz_coords, order=0, output=self.translated_volume)

    def scale(self, factor=None):
        '''
        Scales cylinder along Z axis.
        '''
        if not factor:
            factor = self.psf
        self.scaled_volume = zoom(self.translated_volume, (1, 1, factor), order=1)
        self.scaled_volume[self.scaled_volume < .001] = 0

    def get_bottom(self, top, height, psi=0, theta=0):
        '''
        Gets the other end of the cylinder given a seed and optimized psi and theta. To be used for extending cylinder.
        :param top: Seed point of cylinder to extend
        :psi: Best angle to transform
        :theta: Best angle to transform
        :return: end of correctly oriented   cylinder
        '''

        orig_end = [x + y for x, y in zip(top,[0, height, 0])]
        rot_mat = t3d.euler.euler2mat(np.deg2rad(psi), 0, np.deg2rad(theta), axes='sxyz')
        bottom = np.rint(rot_mat.dot(orig_end.T).T).astype(np.int64)
        return bottom
    ########################################################################################################################

# ** OUTPUT FUNCTIONS ** #

########################################################################################################################

    def render_voxels(self, volume_choice='translated'):

        """
        Renders cylinder [binary mask] in 3D. 'Volume choice' lets you iterate between options for cylinder:
        'original' renders the original mask, while 'translated' renders rotated mask.
        """

        volume = {'original': self.original_volume, 'translated': self.translated_volume}[volume_choice]
        colors = np.empty(volume.shape, dtype=object)
        volume_mask = volume > 0
        colors[volume_mask] = 'red'
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.voxels(volume_mask, facecolors=colors, edgecolor='k')
        plt.show()

########################################################################################################################

    def render_gauss(self, volume_choice='translated'):

        """
        Renders cylinder [Gaussian stack] in 3D. 'Volume choice' lets you iterate between options for cylinder:
        'original' renders the original cylinder, while 'translated' renders rotated cylinder.
        """

        volume = {'original': self.original_volume, 'translated': self.translated_volume}[volume_choice]
        colors = np.zeros(volume.shape + (4,))
        colors[..., 0] = 1
        colors[..., 3] = volume
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.voxels(volume > 0, facecolors=colors)
        plt.show()

########################################################################################################################
# ** MISC ** #
########################################################################################################################

    def get_image_indices(self, seed=None):
        x, y, z = seed

        x_start = x - self.height - self.radius
        x_end = x + self.height + self.radius + 1
        y_start = y - self.height - self.radius
        y_end = y + self.height + self.radius + 1
        z_start = z - self.radius
        z_end = z + self.height + self.radius + 1

        if self.first:
            z_start = z_start - self.height

        return (slice(z_start, z_end), slice(y_start, y_end), slice(x_start, x_end))

