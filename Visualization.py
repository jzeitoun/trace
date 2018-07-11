import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import tifffile as tiff

def overlay_cylinder(filename, img, cylinder, seed_coordinate):
    # Seed coordinate should be in "x,y,z" (this translates to "z,y,x" in numpy
    # array coordinates)

    transposed_cylinder = cylinder.translated_volume.transpose(2,1,0)
    volume = np.zeros(img.shape, dtype='uint8')
    cropped_volume = volume[cylinder.get_image_indices(seed_coordinate)]
    cropped_volume[:] = transposed_cylinder * 255
    overlaid_img = np.stack((volume, img), axis=1)
    tiff.imsave(filename, overlaid_img, imagej=True)

def render_voxels(cylinder, volume_choice='translated'):

    """
    Renders cylinder [binary mask] in 3D. 'Volume choice' lets you iterate between options for cylinder:
    'original' renders the original mask, while 'translated' renders rotated mask.
    """

    volume = {'original': cylinder.original_volume, 'translated': cylinder.translated_volume}[volume_choice]
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



