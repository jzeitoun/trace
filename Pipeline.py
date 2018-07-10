import tifffile as tif
import Seed as s
from Cylinder_Object import Cylinder
import Optimize
import Visualization as vis

import Settings


#def main():
Settings.init()

default_radius = 2
default_height = 5

# Load in image and seeds
filename = 'testdata.tif'
img = tif.imread(filename)
#seedfilename = 'testdata_maxima_binary.tif'
#seeds = s.get_seeds(seedfilename)

cylinder = Cylinder(default_radius, default_height)
vis.render_gauss(cylinder)

#best_score, best_theta, best_psi = Optimize.optimize_angle(cylinder, seeds[257], img)

#fit_score = cylinder._score_correlation(img)

#if __name__ == '__main__':
#   main()
