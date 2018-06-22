import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import tifffile as tiff

def visualise_cylinder(filename, img, Cylinder, seed_coordinate, translated = False):
    # Determines size of output image.
    # Image will have the seed point in the center of the stack.
    # Stack will extend outward in either direction by cropping_factor * height of cylinder.

    cropping_factor = 2
    crop_offset = (cropping_factor * Cylinder.height) + Cylinder.radius

    # Crop image
    cropped_img = img[
                  seed_coordinate[0] - crop_offset : seed_coordinate[0] + crop_offset,
                  seed_coordinate[1] - crop_offset : seed_coordinate[1] + crop_offset,
                  seed_coordinate[2] - crop_offset : seed_coordinate[2] + crop_offset
                  ]
    volume = np.zeros(cropped_img.shape)

    if translated == True:
        indices_coordinates = Cylinder.translated_indices - Cylinder.original_center
        indices_coordinates = indices_coordinates + crop_offset

        for i in range(indices_coordinates.shape[0]):
            volume[indices_coordinates[i][0]][indices_coordinates[i][1]][indices_coordinates[i][2]] = \
            Cylinder.translated_values[i] * 255

    if translated == False:
        indices_coordinates = Cylinder.original_indices - Cylinder.original_center
        indices_coordinates = indices_coordinates + crop_offset

        for i in range(indices_coordinates.shape[0]):
            volume[indices_coordinates[i][0]][indices_coordinates[i][1]][indices_coordinates[i][2]] = \
            Cylinder.original_values[i] * 255

    volume.shape = volume.shape + (1,)
    cropped_img.shape = cropped_img.shape + (1,)
    a = np.concatenate((volume, cropped_img), axis=3)
    tiff.imsave(filename, a.transpose([0, 1, 2, 3]), photometric='minisblack', planarconfig='contig', bigtiff=True)

def render_voxels(Cylinder, volume_choice='translated'):

    """
    Renders cylinder [binary mask] in 3D. 'Volume choice' lets you iterate between options for cylinder:
    'original' renders the original mask, while 'translated' renders rotated mask.
    """

    volume = {'original': Cylinder.original_volume, 'translated': Cylinder.translated_volume}[volume_choice]
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



