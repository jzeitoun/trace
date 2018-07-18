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


cylinder = Cylinder(default_radius, default_height, psf=2, first=True)

# Correct indices (if negative) and pad cropped image if necessary
indices = cylinder.get_image_indices(seed)
corrected_indices, pad_sequence = check_bounds(indices, whole_img)

cropped_img = whole_img[corrected_indices]
padded_cropped_img = np.pad(cropped_img, pad_sequence, mode='constant')
score, best_theta, best_psi = opt.optimize_angle(cylinder, seed, padded_cropped_img)
cylinder.rotate(best_psi, best_theta)

vis.overlay_cylinder('output.tif', whole_img, cylinder, seed)

#best_score, best_theta, best_psi = Optimize.optimize_angle(cylinder, seeds[257], img)

#fit_score = cylinder._score_correlation(img)

#if __name__ == '__main__':
#   main()
