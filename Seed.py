import numpy as np
import tifffile as tif
import Math as math


#######################################################################################################################

def get_seeds(seedfilename):
    """
    Returns an N x 3 array of coordinates of seeds. Local maxima is currently arbitrarily calculated in Fiji.
    Calls read_image()
    *Temporary utility. Must be changed.*

        Input:
            <string> seedfilename [binary image of local maxima points]

        Output:
            <int array [2D]> seeds [seed coordinates]
    """
    seed_img = tif.imread(seedfilename)
    seeds = np.argwhere(seed_img)  # returns those coordinates where the seed exists [value = 1].

    return seeds


#######################################################################################################################

def make_seed_encyclopedia(total_seeds):
    seed_encyclopedia = [{'Seed_Coordinates': 0,
                          'Cylinder_Radius': [],
                          'Cylinder_Height': [],
                          'Cylinder_Theta': [],
                          'Cylinder_Psi': [],
                          'Fit_Cylinder_Coordinates': [],
                          'Number_of_Cylinders': 0,
                          'Bottom_Reference': 0}
                         for k in range(total_seeds)]  # making data type that stores info on all seeds.

    return seed_encyclopedia


#######################################################################################################################

def update_seed_encyclopedia(index, seed_coordinates, cylinder, bottom_reference, seed_encyclopedia):
    seed_encyclopedia[index]['Seed_Coordinates'].append(seed_coordinates)
    seed_encyclopedia[index]['Cylinder_Radius'].append(cylinder.radius)
    seed_encyclopedia[index]['Cylinder_Height'].append(cylinder.height)
    seed_encyclopedia[index]['Cylinder_Theta'].append(cylinder.theta)
    seed_encyclopedia[index]['Cylinder_Psi'].append(cylinder.psi)
    seed_encyclopedia[index]['Bottom_Reference'].append(bottom_reference)
    seed_encyclopedia[index]['Number_of_Cylinders'] += 1
    n = seed_encyclopedia[index]['Number_of_Cylinders']
    seed_encyclopedia[index]['Fit_Cylinder_Coordinates'][:, n] = cylinder.translated_volume

    return seed_encyclopedia


########################################################################################################################

def make_seed_score_directory(total_seeds):
    seed_score_directory = [{'Best_Position_Score_1': [],
                             'Best_Position_Score_2': [],
                             'Best_Angle_Score': [],
                             'Best_Radius_Score': []}
                            for k in range(total_seeds)]

    return seed_score_directory


########################################################################################################################

def update_seed_score_directory(index, position_score_1, position_score_2, angle_score, radius_score,
                                seed_score_directory):
    seed_score_directory[index]['Best_Position_Score_1'].append(position_score_1)
    seed_score_directory[index]['Best_Position_Score_2'].append(position_score_2)
    seed_score_directory[index]['Best_Angle_Score'].append(angle_score)
    seed_score_directory[index]['Best_Radius_Score'].append(radius_score)

    return seed_score_directory


########################################################################################################################





"""def main_call(seeds, img):

     *INCOMPLETE*
    Traverses through seed coordinates and calls cylinder_utils to perform cylinder fitting.
    seed_encyclopedia is initialised and updated.

    total_seeds = seeds.shape[0]



    def_radius = 2 #radius must be even for mask to work. *must be tweaked*
    def_height = 3 #will be made dynamic? Just for first drop.

    for i, seed_i in range(total_seeds), seed_encyclopedia:
        seed_x = seeds[i,0]
        seed_y = seeds[i,1]
        seed_z = seeds[i,2]
        cylinder_mask, cylinder_coord = cyl.make_cylinder(seed_x, seed_y, seed_z, def_radius, def_height)
        io.render(cylinder_mask) #for testing purposes only. Deactivate before compilation.
        cylinder_mask, cylinder_coord = cyl.optimize_angle(seeds[i,:], img, cylinder_mask)
        io.render(cylinder_mask)  # for testing purposes only. Deactivate before compilation.
        seed_encyclopedia[seed_i] = update_encyclopedia(seed_encyclopedia, seed_i, seeds[i,:], cylinder_mask, cylinder_coord)

       ]

    return seed_encyclopedia
def update_encyclopedia(seed, seed_index, seeds, cylinder_mask, cylinder_coord):
    # Update encylopedia:
    seed['Index'] = seed_index
    seed['Coordinates'] = seeds[i, :]
    seed['Fit_Cylinder_Mask'] = cylinder_mask
    seed['Fit_Cylinder_Coordinates'] = cylinder_coord
    seed['Number_of_Cylinders'] += 1
    seed['Bottom_Reference'] = seeds[i, :]

    return seed"""


def filter_seeds(seeds):
    """
    Removes all seeds that do not belong on a 'line', by calculating the hessian of each point followed by the
    eigen value of the resulting hessian. A hessian matrix here is a 3 x 3 matrix of the second partial derivatives
    along the x, y, and z axes. It is done by hessian() defined in math_utils.py

    A seed is concluded to be "on a line" by the nature of its eigen values lambda1, lambda2, and lambda3:
    one of them must be equal to 0 while the other 2 must be roughly equal to each other and lesser than 0.

        Input:
            <N x 3> array of raw seed coordinates.

        Output:
            <N x 3> array of seed coordinates that supposedly fall on a line.

    """
    total_seeds = seeds.shape[0]
    filtered_seeds = []
    for i in range(total_seeds):
        hess_seed = math.hessian(seeds[i, :])
        eigval_hess, eigvec_hess = np.linalg.eig(hess_seed)
        if (eigval_hess[0] == 0) & (eigval_hess[1] == eigval_hess[2]) & (eigval_hess[1] > 0) & (eigval_hess[2] > 0):
            filtered_seeds = np.append(filtered_seeds, seeds[i, :])

    return filtered_seeds

    #######################################################################################################################