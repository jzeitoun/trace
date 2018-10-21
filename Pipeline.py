import numpy as np
import tifffile as tif

import Seed as s
from Cylinder_Object import Cylinder
import Optimize as opt
import Visualization as vis
from Util import check_bounds

import Settings


#def main():
Settings.init()

default_radius = 4
default_height = 10

# Load in image and seeds
filename = 'testdata.tif'
whole_img = tif.imread(filename)
#seedfilename = 'testdata_maxima_binary.tif'
#seeds = s.get_seeds(seedfilename)
#seed = [37, 39, 7]
#seed = [42, 48, 7]
#seed = [46, 57, 7]
#seed = [53, 84, 5]
seed = [63, 96, 1]

# Create default cylinder
cylinder = Cylinder(default_radius, default_height, 0, 0, psf=2, first=True)

# Correct indices (if negative) and pad cropped image if necessary
indices = cylinder.get_image_indices(seed)
corrected_indices, pad_sequence = check_bounds(indices, whole_img)

cropped_img = whole_img[corrected_indices]
padded_cropped_img = np.pad(cropped_img, pad_sequence, mode='constant')

# Loop through  all seeds here.
# Extend out from a single seed
seed_flag = 0  # update to 1 to stop extending
while seed_flag != 1:
    # Optimizing angle:
    score, best_theta, best_psi = opt.optimize_angle(cylinder, seed, padded_cropped_img)
    cylinder.rotate(best_psi, best_theta)

    # Optimizing radius:
    best_radius = opt.optimize_radius(cylinder, seed, padded_cropped_img)
    cylinder = Cylinder(default_radius, default_height, best_psi, best_theta, psf=2, first=True)
    vis.overlay_cylinder('output.tif', whole_img, cylinder, seed)

    # Optimize height
    best_height = default_height

    # Repeating:
    seed = cylinder.get_bottom(seed, best_height, best_psi, best_theta)

#if __name__ == '__main__':
#   main()
