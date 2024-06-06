import logging
import cv2
import numpy as np
import math
from joblib import Parallel, delayed
from cmaas_utils.types import Legend

from .utils import boundingBox

# Plot effects
CORRECT_COLOR = (64,255,64) # Green
FAIL_COLOR = (255,1,1) # Red
MISS_COLOR = (255,1,255) # Fuchsia

log = logging.getLogger('DARPA_CMAAS_VALIDATION')

def grade_poly_raster(pred_image, true_image, feedback_image=None):
    # Do we need to convert to binary mask? Currently i don't think so
    intersection = pred_image & true_image
    union = pred_image | true_image

    true_positive = np.count_nonzero(intersection)
    # avoid div by 0
    if true_positive == 0:
        result = {'F1 Score' : 0, 'Precision' : 0, 'Recall' : 0, 'IoU Score' : 0}
        return result, feedback_image

    result = {}
    recall = true_positive / np.count_nonzero(true_image)
    precision = true_positive / np.count_nonzero(pred_image)
    
    result['Recall'] = recall
    result['Precision'] = precision
    result['F1 Score'] = 2 * ((precision * recall)/(precision + recall))
    result['IoU Score'] = true_positive / np.count_nonzero(union)

    if feedback_image is not None:
        feedback_image[0][(true_image>=1)[0]] = MISS_COLOR[0]
        feedback_image[1][(true_image>=1)[0]] = MISS_COLOR[1]
        feedback_image[2][(true_image>=1)[0]] = MISS_COLOR[2]
        feedback_image[0][(pred_image>=1)[0]] = FAIL_COLOR[0]
        feedback_image[1][(pred_image>=1)[0]] = FAIL_COLOR[1]
        feedback_image[2][(pred_image>=1)[0]] = FAIL_COLOR[2]
        feedback_image[0][(intersection==1)[0]] = CORRECT_COLOR[0]
        feedback_image[1][(intersection==1)[0]] = CORRECT_COLOR[1]
        feedback_image[2][(intersection==1)[0]] = CORRECT_COLOR[2]

    return result, feedback_image

def grade_point_raster(pred_image, true_image, feedback_image=None, min_valid_range=0.1):
    """
    pred_image : 2d np array, with 0s and 1s only
    true_image : 2d np array, with 0s and 1s only
    feedback_image : 3d np array, image to draw feedback on
    min_valid_range : the maximum distance in % of the largest size of the image (diagonal)
        between a predicted pixel vs. a true one that will be considered
        as valid to include in the scoring.
    returns : Dictionary of scores, feedback image

    Changes from USGS version:
        This function contains the point grading portion of the USGS function "feature_f_score"
        Added a debuging image
        Changed np.sum(pred_image) to np.count_nonzero(pred_image) for precision / recall scores. Shouldn't change result but is more explicit in what is actually happening
        Changed all the if var!=0 else 0.0 to a single if len(match_pt_pairs) check at the start of the function
    """
    matched_pt_pairs = match_nearest_points(true_image, pred_image, min_valid_range=min_valid_range)
    # Check if there were any matched points
    if (len(matched_pt_pairs) == 0):
        result = {
            'F1 Score' : 0, 
            'Precision' : 0, 
            'Recall' : 0,
            'Mean Matched Distance' : np.nan, 
            'Matched Points' : 0,
            'Missing Points' : np.count_nonzero(true_image), 
            'Unmatched Points' : np.count_nonzero(pred_image)
        }
        return result, feedback_image

    result = {}

    # Get the mean distance
    norm_mean_dist = sum([p[1] for p in matched_pt_pairs]) / len(matched_pt_pairs)
    map_size = math.sqrt(math.pow(true_image.shape[0], 2) + math.pow(true_image.shape[1], 2))
    result['Mean Matched Distance'] = norm_mean_dist * map_size
    result['Matched Points'] = len(matched_pt_pairs) # Green
    result['Missing Points'] = np.count_nonzero(true_image) - result['Matched Points'] # Pink
    result['Unmatched Points'] = np.count_nonzero(pred_image) - result['Matched Points'] # Red

    # Calculate statistical values
    sum_of_similarities = sum([1-item[1] for item in matched_pt_pairs])
    result['Precision'] = sum_of_similarities / np.count_nonzero(pred_image)
    result['Recall'] = sum_of_similarities / np.count_nonzero(true_image)
    result['F1 Score'] = (2 * result['Precision'] * result['Recall']) / (result['Precision'] + result['Recall'])

    # Draw image feedback
    if feedback_image is not None:
        pt_radius = math.floor(map_size * 0.001)
        thickness = 5
        used_pts = set()
        # Correct pts
        for p in matched_pt_pairs:
            feedback_image = cv2.circle(feedback_image, (p[0][0][1],p[0][0][0]), pt_radius, CORRECT_COLOR, thickness)
            used_pts.add(p[0][0])
            used_pts.add(p[0][1])
        # Missing pts
        for x, y, _ in np.argwhere(true_image==1):
            if (x,y) in used_pts:
                continue
            feedback_image = cv2.circle(feedback_image, (y,x), pt_radius, MISS_COLOR, thickness)
        # Unmatched pts
        for x, y, _ in np.argwhere(pred_image==1):
            if (x,y) in used_pts:
                continue
            feedback_image = cv2.circle(feedback_image, (y,x), pt_radius, FAIL_COLOR, thickness)

    return result, feedback_image

def match_nearest_points(true_image, pred_image, min_valid_range=0.1, parallel_workers=1):
    """
    true_image : 2d np array, with 0s and 1s only
    pred_image : 2d np array, with 0s and 1s only
    min_valid_range: the maximum distance in % of the largest size of the image (diagonal)
        between a predicted pixel vs. a true one that will be considered
        as valid to include in the scoring.
    parallel_workers: amount of processes to use for calculating the distance metrics

    Changes from USGS version:
        Changed function name from "overlap_distance_calculate" to "match_nearest_points"
        Changed function parameter names and updated docstring
        Changed internal function variable names to be easier to read
        Changed formatting to be easier to read (Made sure things were spaced out properly)
        Removed tqdm progress bars
        Removed print statements
    """
    pred_image = np.squeeze(pred_image)
    true_image = np.squeeze(true_image)
    
    lowest_dist_pairs = []
    pred_points_done = set()
    true_points_done = set()

    # perfect prediction pixels
    intersection = pred_image*true_image
    for x, y in np.argwhere(intersection>=1):
        lowest_dist_pairs.append((((x, y), (x, y)), 0.0)) 
        true_points_done.add((x, y))
        pred_points_done.add((x, y))
    
    diagonal_length = math.sqrt(math.pow(true_image.shape[0], 2) + math.pow(true_image.shape[1], 2))
    min_valid_range = int((min_valid_range*diagonal_length)/100) # in pixels

    def nearest_pixels(x, y):
        result=[]
        # find all the points in pred withing min_valid_range rectangle
        mat_pred_inrange = pred_image[
            max(x-min_valid_range, 0): min(x+min_valid_range, true_image.shape[0]),
            max(y-min_valid_range, 0): min(y+min_valid_range, true_image.shape[1])]
        for x_pred_shift, y_pred_shift in np.argwhere(mat_pred_inrange>=1):
            y_pred = max(y-min_valid_range, 0) + y_pred_shift
            x_pred = max(x-min_valid_range, 0) + x_pred_shift
            if (x_pred, y_pred) in pred_points_done:
                continue
            # calculate eucledean distances 
            dist_square = math.pow(x-x_pred, 2) + math.pow(y-y_pred, 2)
            result.append((((x, y), (x_pred, y_pred)), dist_square))
        return result

    candidates = [(x, y) for x, y in np.argwhere(true_image>=1) if (x, y) not in true_points_done]
    distances = Parallel(n_jobs=parallel_workers)(delayed(nearest_pixels)(x, y) for x, y in candidates)
    distances = [item for sublist in distances for item in sublist]

    # sort based on distances
    distances = sorted(distances, key=lambda x: x[1])

    # find the lowest distance pairs
    for ((x, y), (x_pred, y_pred)), distance in distances:
        if ((x, y) in true_points_done) or ((x_pred, y_pred) in pred_points_done):
            # do not consider a taken point again
            continue
        # normalize all distances by diving by the diagonal length  
        lowest_dist_pairs.append((((x, y), (x_pred, y_pred)), math.sqrt(float(distance))/diagonal_length)) 
        true_points_done.add((x, y))
        pred_points_done.add((x_pred, y_pred))
    
    return lowest_dist_pairs

def usgs_grade_poly_raster(pred_image, true_image, image, feature_bounding_box, feedback_image=None, difficult_weight=0.7, color_range=4, set_false_as='hard'):
    """
    pred_image : 2d np array, with 0s and 1s only
    true_image : 2d np array, with 0s and 1s only
    image : 3d np array of the map image
    feature_bounding_box : 
    feedback_image : 3d np array, image to draw feedback on
    difficult_weight : float within [0, 1], weight for the difficult pixels in the scores
    color_range : the range of color variation to consider for the legend color
    set_false_as : when set to 'hard' the pixels that are not within the true polygon area will be considered hard

    Changes from USGS version:
        This function contains the point grading portion of the USGS function "feature_f_score"
        Added a debuging image
        Changed np.sum(pred_image) to np.count_nonzero(pred_image) for precision / recall scores. Shouldn't change result but is more explicit in what is actually happening
        Changed all the if var!=0 else 0.0 to a single if true_positive == 0 check at the start of the function
        Removed print statements
    """
    intersection = true_image * pred_image
    union = true_image | pred_image
    true_positive = np.count_nonzero(intersection)

    if true_positive == 0:
        return {'F1 Score' : 0, 'Precision' : 0, 'Recall' : 0, 'IoU Score' : 0}, feedback_image

    if difficult_weight is None:
        precision = true_positive/np.count_nonzero(pred_image)
        recall = true_positive/np.count_nonzero(true_image)

    else:
        hard_pixel_mask = detect_difficult_pixels(image, true_image, feature_bounding_box=feature_bounding_box, set_false_as=set_false_as, color_range=color_range)
            
        ### Weighted Intersection
        intersection_hard = intersection*hard_pixel_mask
        true_positive_hard = np.count_nonzero(intersection_hard)
        true_positive_easy = np.count_nonzero(intersection-intersection_hard)
        true_positive_weighted = (true_positive_hard*difficult_weight) + (true_positive_easy*(1-difficult_weight))

        ### Weighted Precision
        predicted_hard = np.count_nonzero(pred_image*hard_pixel_mask)
        predicted_easy = np.count_nonzero(pred_image-(pred_image*hard_pixel_mask))
        total_pred = (predicted_hard*difficult_weight) + (predicted_easy*(1-difficult_weight))
        precision = true_positive_weighted/total_pred

        ### Weighted Recall 
        true_hard = len(np.argwhere((true_image*hard_pixel_mask)==1))
        true_easy = len(np.argwhere((true_image-(true_image*hard_pixel_mask))==1))
        total_true = (true_hard*difficult_weight)+(true_easy*(1-difficult_weight))
        recall = true_positive_weighted/total_true
    
    result = {}
    result['Recall'] = recall
    result['Precision'] = precision
    result['F1 Score'] = (2 * precision * recall)/(precision+recall)
    result['IoU Score'] = true_positive / np.count_nonzero(union)

    if feedback_image is not None:
        feedback_image[(true_image>=1).all(-1)] = MISS_COLOR
        feedback_image[(image>=1).all(-1)] = FAIL_COLOR
        feedback_image[(intersection==1).all(-1)] = CORRECT_COLOR

    return result, feedback_image

def match_by_color(map_image, feature_bounding_box, color_range=4):
    """
    map_image: Numpy array of the map image, expected format is (CHW)
    feature_bounding_box: coordinates for the legend feature, expected format is [[min_x, min_y], [max_x, max_y]]
    color_range: the range of color variation to consider for the legend color

    Changes from USGS version:
        Replaced there method for finding the legend bounding box with ours
        Changed function parameter names and updated docstring
        Changed internal function variable names to be easier to read
        Changed formatting to be easier to read (Made sure things were spaced out properly)
        Removed tqdm progress bars
        Removed print statements
    """
    # get the legend coors and the predominant color
    min_pt, max_pt = boundingBox(feature_bounding_box)      
    lgd_image = map_image[:, int(min_pt[1]):int(max_pt[1]), int(min_pt[0]):int(max_pt[0])]

    # take the median of the colors to find the predominant color
    median_color = [np.median(lgd_image[0,:,:]), np.median(lgd_image[1,:,:]), np.median(lgd_image[2,:,:])]

    # capture the variations of legend color due to scanning errors
    lower = np.array(median_color) - color_range
    lower[lower<0] = 0
    upper = np.array(median_color) + color_range
    upper[upper>255] = 255

    # create a mask to only preserve current legend color in the basemap
    map_image = map_image.transpose(1, 2, 0)
    mask = cv2.inRange(map_image, lower, upper) / 255

    return mask

def detect_difficult_pixels(map_image, true_image, feature_bounding_box, color_range=4, set_false_as='hard'):
    """
    map_image: the image array for the map image
    true_image: 2D array of any channel (out of 3 present) from the true binary raster image 
    lgd_pts: coordinate for the legend feature, from the legend json file
    set_false_as: when set to 'hard' the pixels that are not within the true polygon area will be considered hard

    Changes from USGS version:
        Changed function parameter names and updated docstring
        Removed print statements
    """
    true_image = np.squeeze(true_image)
        
    # detect pixels based on color of legend
    pred_by_color=match_by_color(map_image, feature_bounding_box, color_range=color_range)
            
    pred_by_color=(1-pred_by_color).astype(np.uint8) # flip, so the unpredicted become hard pixels
    pred_by_color=true_image*pred_by_color # keep only the part within the true polygon
    
    if set_false_as=='hard':
        # the pixels that are not within the true polygon should are deemed hard pixels
        hard_pixel_mask=(1-true_image) | pred_by_color
    else:
        # the outside pixels will be deemed easy!
        hard_pixel_mask=pred_by_color

    return hard_pixel_mask
