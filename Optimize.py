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
########################################################################################################################

def optimize_angle(cylinder, seed, raw_img):
    img = raw_img.transpose(2,1,0)
    # Calculating step size:
    #import ipdb; ipdb.set_trace()
    step_dec = math.degrees(2*np.arcsin(cylinder.radius/(2*cylinder.height)))
    mod90 = 90 % step_dec
    mod360 = 360 % step_dec
    div90 = 90//step_dec
    div360 = 360//step_dec
    psi_step = 5#step_dec + mod90/div90
    theta_step = 5#step_dec + mod360/div360

    fit_score = []
    psi_arr = []
    theta_arr = []
    psi = 0

    psi_max = 180 if cylinder.first else 90

    for psi in range(0, psi_max, psi_step):
        for theta in range(0, 360, theta_step):
            cylinder.rotate(psi, theta)
            score = score_fit(cylinder, img)
            fit_score.append(score)
            theta_arr.append(theta)
            psi_arr.append(psi)

    best_score = max(fit_score)
    high_score_index = fit_score.index(best_score)
    theta = theta_arr[high_score_index]
    psi = psi_arr[high_score_index]

    #cylinder.rotate(theta, psi)

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

def score_fit(cylinder, cropped_img):
    """
    Calculates correlation coefficient between original image and cylinder.

    Returns:
        Correlation score.

    """
    cylinder_indices = cylinder.translated_coords
    img_values = cropped_img[[*cylinder_indices.T]]
    sdA = np.abs(img_values - np.mean(img_values))
    sdB = np.abs(cylinder.translated_values - np.mean(cylinder.translated_values))
    reg = np.sum(sdA * sdB)
    return reg / np.sqrt(np.sum(sdA ** 2) * (np.sum(sdB ** 2)))
