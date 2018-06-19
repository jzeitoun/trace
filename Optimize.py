import numpy as np
import math
import Cylinder_Object as cyl
import Visualization as vis

########################################################################################################################
# ** OPTIMIZATION FUNCTIONS ** #
########################################################################################################################

def optimize_cylinder(cylinder, seed, img):
    """
    Optimizes the cylinder fit along the inputted seed point.
    It first performs a position adjust on the seed point to get new seed point coordinates.
    It then performs an orientation adjust (from the new coordinates), after which it repeats its position adjust to retain optimum position.
    It then fine tunes the fit by optimizing the radius.

    Inputs:
        <object> cylinder: The default fitted cylinder.
        <1 x 3 array> seed: The coordinates of the specified seed point.
        <X x Y x Z array> img: The original image file

    Outputs:
        best_score_pos1: The best fit score outputted from the first position adjust loop
        best_score_pos2: The best fit score outputted from the second position adjust loop
        best_score_ang: The best fit score outputted from the orientation adjust loop
        best_score_rad: The best fit score outputted from the radius adjust loop.
        new_seed: The resulting coordinates for seed from the two position adjust loops.
        best_rad, best_theta, best_psi: The best radii and angles respectively.
        cylinder.translated_volume: The new cylinder positions as a result of the orientation fit.


    """
    best_score_pos1, new_seed = optimize_position(cylinder, seed, img)
    best_score_ang, best_theta, best_psi = optimize_angle(cylinder, new_seed, img)
    best_score_pos2, new_seed = optimize_position(cylinder, new_seed, img)

    best_score_rad, best_rad = optimize_radius(cylinder, new_seed, img)
    cylinder = cyl.Cylinder(best_rad)
    cyl.rotate(best_theta, best_psi)

    return best_score_pos1, best_score_pos2, best_score_ang, best_score_rad, new_seed, cylinder


def optimize_position(cylinder, seed, img):

    r = cylinder.radius

# Getting minimum and maximum search values in all three directions:
    if seed[0] >= r:
        offset_x = seed[0]-r
    else:
        offset_x = seed[0]

    if seed[1] >= r:
        offset_y = seed[1]-r
    else:
        offset_y = seed[1]

    if seed[2] >= r:
        offset_z = seed[2]-r
    else:
        offset_z = seed[2]

    if seed[0] <= [img.shape[0]-r]:
        max_x = seed[0] + r
    else:
        max_x = seed[0]

    if seed[1] <= [img.shape[1] - r]:
        max_y = seed[1] + r
    else:
        max_y = seed[1]

    if seed[2] <= [img.shape[2] - r]:
        max_z = seed[2] + r
    else:
        max_z = seed[2]

# Initializing arrays and step size:
    step = 1
    fit_score = []
    center_x_arr = []
    center_y_arr = []
    center_z_arr = []

# Loop for optimizing position:
    while offset_x <= max_x and offset_y <= max_y and offset_z <= max_z:
        offset = [offset_x, offset_y, offset_z]
        new_seed = seed + offset # get position of new seed in loop
        cropped_img = seed_cropped_img(cylinder, new_seed, img) # generate cropped image from original image with 'new_seed' as center
        score = score_fit(cylinder, cylinder.original_indices, cropped_img) # calculate fit score between cylinder and cropped image

        fit_score.append(score)
        center_x_arr.append(offset_x)
        center_y_arr.append(offset_y)
        center_z_arr.append(offset_z)

        offset_x, offset_y, offset_z = offset_x + step, offset_y + step, offset_z + step #update step size

# Getting best score:
    best_score = max(fit_score)
    high_score_index = fit_score.index(best_score)
    new_seed = [center_x_arr[high_score_index], center_y_arr[high_score_index], center_z_arr[high_score_index]]

    return best_score, new_seed

########################################################################################################################

def optimize_angle(cylinder, seed, img):

    # Calculating step size based on radius and height:
    step_dec = math.degrees(2*np.arcsin(cylinder.radius/(2*cylinder.height)))
    mod90 = 90 % step_dec
    mod360 = 360 % step_dec
    div90 = 90//step_dec
    div360 = 360//step_dec
    step_psi = step_dec + mod90/div90
    step_theta = step_dec + mod360/div360

    fit_score = []
    theta_arr = []
    psi_arr = []
    psi= 0

    while psi <= 90:
        theta = 0

        if psi == 0:  # Don't need to rotate if vertical
            cropped_img = mask_dataset(cylinder, seed, img)
            score = score_fit(cylinder, cropped_img, translated= False)
            vis.visualise_cylinder(str(score) + ' theta ' + str(theta) + ' psi ' + str(psi) + '.tif', img, cylinder,
                                   seed, translated= False)
            fit_score.append(score)
            theta_arr.append(theta)
            psi_arr.append(psi)
        else:
            while theta < 360:
                cylinder.rotate(theta, psi)
                cropped_img = mask_dataset(cylinder, seed, img)
                score = score_fit(cylinder, cropped_img, translated = True)
                vis.visualise_cylinder(str(score) + ' theta' + str(theta) + ' psi ' + str(psi) + '.tif', img, cylinder,
                                       seed, translated= True)

                fit_score.append(score)
                theta_arr.append(theta)
                psi_arr.append(psi)

                theta = step_theta + theta
        psi = step_psi + psi

    best_score = max(fit_score)
    high_score_index = fit_score.index(best_score)
    theta = theta_arr[high_score_index]
    psi = psi_arr[high_score_index]

    cylinder.rotate(theta, psi)

    return best_score, theta, psi

########################################################################################################################

def optimize_radius(cylinder, seed, img):

    h = cylinder.height
    r = 1
    r_step = 1
    max_r = 5
    fit_score = []
    rad_arr = []

    while r < max_r:

        cylinder(r, h)
        cropped_img = seed_cropped_img(cylinder, seed, img)
        score = score_fit(cylinder, cylinder.original_indices, cropped_img)
        fit_score.append(score)
        rad_arr.append(r)
        r = r+r_step

    best_score = max(fit_score)
    high_score_index = fit_score.index(best_score)
    best_rad = rad_arr[high_score_index]

    return best_score, best_rad

########################################################################################################################

def score_fit(cylinder, cropped_img, translated = True):
    """
    Calculates correlation coefficient between original image and cylinder.

    Returns:
        Correlation score.

    """

    if translated == True:
        img_values = cropped_img[[*cylinder.translated_indices.T]]
        sdA = np.abs(img_values - np.mean(img_values))
        sdB = np.abs(cylinder.translated_values - np.mean(cylinder.translated_values))
    else:
        img_values = cropped_img[[*cylinder.original_indices.T]]
        sdA = np.abs(img_values - np.mean(img_values))
        sdB = np.abs(cylinder.original_values - np.mean(cylinder.original_values))
    reg = np.sum(sdA * sdB)
    return reg / np.sqrt(np.sum(sdA ** 2) * (np.sum(sdB ** 2)))

########################################################################################################################

def mask_dataset(cylinder, ref_point, img):

    """
    Returns corresponding image, to be considered, given cylinder dimensions and desired seed point.
    """
    #AXES ARE INCORRECT
    #IMG IS Z,Y,X
    z, y, x = ref_point
    if x != 0:
        xs = slice(x - cylinder.original_center[0], x + cylinder.original_center[0] + 1)
    else:
        xs = slice(0, x + cylinder.original_center[0] + 1)

    if y != 0:
        ys = slice(y - cylinder.original_center[1], y + cylinder.original_center[1] + 1)
    else:
        ys = slice(0, y + cylinder.original_center[1] + 1)

    if z != 0:
        zs = slice(z - cylinder.original_center[2], z + cylinder.original_center[2] + cylinder.height)
    else:
        zs = slice(0, z + cylinder.radius + cylinder.original_center[2])

    ind = [zs, ys, xs]

    return img[ind]

