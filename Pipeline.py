import tifffile as tif
import Seed as s
from Cylinder_Object import Cylinder
import Optimize as opt
import Visualization as vis

import Settings


#def main():
Settings.init()

default_radius = 4
default_height = 12

# Load in image and seeds
filename = 'testdata.tif'
whole_img = tif.imread(filename)
#seedfilename = 'testdata_maxima_binary.tif'
#seeds = s.get_seeds(seedfilename)
seed = [37, 39, 7]

cylinder = Cylinder(default_radius, default_height)
#vis.render_gauss(cylinder)

img = whole_img[cylinder.get_image_indices(seed)]
score, best_theta, best_psi = opt.optimize_angle(cylinder, seed, img)
cylinder.rotate(best_psi, best_theta)

vis.overlay_cylinder('output.tif', whole_img, cylinder, seed)

#best_score, best_theta, best_psi = Optimize.optimize_angle(cylinder, seeds[257], img)

#fit_score = cylinder._score_correlation(img)

#if __name__ == '__main__':
#   main()
