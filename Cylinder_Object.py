import numpy as np
import Math as math
import transforms3d as t3d

class Cylinder(object):

    def __init__(self, radius=1, height=5):
        self.radius = radius
        self.radius_top = radius
        self.height = height
        self.center = None
        self.psi = 0 #angle around z
        self.theta = 0 #angle around x
        self.original_volume = np.array(self._make_gaussian_cylinder(radius=radius, height=height)) #cylinder with radius and height where intensities are Gaussian around center of top of cylinder.
        self.original_center = [height+radius, height+radius, radius]
        self.original_indices = np.argwhere(self.original_volume) #returns coordinates where cylinder exists. An N x 3 array.
        self.original_values = self.original_volume[[*self.original_indices.T]]
        self.translated_volume = np.zeros(self.original_volume.shape) #initialization for rotation

########################################################################################################################
    @property
    def translated_indices(self):
        return np.argwhere(self.translated_volume > 0)

    @property
    def translated_values(self):
        return self.translated_volume[[*self.translated_indices.T]]

########################################################################################################################
# ** BASIC CYLINDER FUNCTIONS: CREATING AND ROTATING ** #
########################################################################################################################

    def _make_cylinder_mask(self, radius, height):
        """Returns a cylinder in a 3d array of Booleans. Array values of 'True' make up the volume of the cylinder.
          Execution time with default params: 1.46ms
          Added radius padding to allow for edge of cylinder transforming beyond height.
        """

        #includes 0 as a value
        x, y, z = np.mgrid[ -height - radius:height + radius + 1, -height - radius:height + radius + 1, -radius:height + radius]
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
        #generate cylinder
        cylinder_mask = self._make_cylinder_mask(radius, height)
        mask_length, mask_width, mask_height = cylinder_mask.shape
        #generate gaussian profile cylinder
        gaussian_kernel = self._make_gaussian(mask_length, radius, center)
        gaussian_cylinder = np.stack([gaussian_kernel] * mask_height, axis=2)
        #mask out values outside cylinder mask by setting them to 0
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

########################################################################################################################

    def move(self, offset):

        moved_indices = self.original_indices + offset
        return moved_indices



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
# ** SCORING FUNCTIONS ** #
########################################################################################################################

    def _score_correlation(self, new_indices, new_values, img):

        """
        Calculates correlation coefficient between original image and translated image.

        Returns:
            Correlation score.

        """
        img_coords = img[new_indices]
        sdA = np.abs(new_values - np.mean(new_values))
        sdB = np.abs(img[img_coords] - np.mean(img[img_coords]))
        reg = np.sum(sdA * sdB)
        return reg / np.sqrt(np.sum(sdA ** 2) * (np.sum(sdB ** 2)))


########################################################################################################################
# ** OPTIMIZATION FUNCTIONS ** #
#

########################################################################################################################
# ** MISC ** #
########################################################################################################################

    def get_image_index(self, indices, seed = None):
        img_coordinates = indices


