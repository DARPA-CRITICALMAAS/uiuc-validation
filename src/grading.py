import logging
import cv2
import numpy as np
import math
from joblib import Parallel, delayed

from .utils import boundingBox

# Plot effects
CORRECT_COLOR = (64,255,64) # Green
FAIL_COLOR = (255,1,1) # Red
MISS_COLOR = (255,1,255) # Fuchsia

#def grade_pt_raster(image, true_image):

log = logging.getLogger('DARPA_CMAAS_VALIDATION')

def grade_poly_raster(pred_image, true_image, feedback_image=None):
    # Do we need to convert to binary mask? Currently i don't think so
    intersection = pred_image & true_image
    union = pred_image | true_image

    true_positive = np.count_nonzero(intersection)
    # avoid div by 0
    if true_positive == 0:
        return (0, 0, 0, 0, feedback_image)

    if feedback_image is not None:
        feedback_image[(true_image>=1).all(-1)] = MISS_COLOR
        feedback_image[(pred_image>=1).all(-1)] = FAIL_COLOR
        feedback_image[(intersection==1).all(-1)] = CORRECT_COLOR

    recall = true_positive / np.count_nonzero(true_image)
    precision = true_positive / np.count_nonzero(pred_image)
    
    f1_score = 2 * ((precision * recall)/(precision+recall))
    iou_score = true_positive / np.count_nonzero(union)

    return (f1_score, precision, recall, iou_score, feedback_image)

def usgs_grade_pt_raster(pred_image, true_image, feedback_image=None):
    matched_pt_pairs=match_nearest_points(true_image, pred_image, min_valid_range=0.25)
    # Check if there were any matched points
    if (len(matched_pt_pairs) == 0):
        return (0,0,0,np.nan,0,np.count_nonzero(true_image),np.count_nonzero(pred_image),feedback_image)
    
    # Get the mean distance
    norm_mean_dist = sum([p[1] for p in matched_pt_pairs]) / len(matched_pt_pairs)
    map_size = math.sqrt(math.pow(true_image.shape[0], 2) + math.pow(true_image.shape[1], 2))
    mean_matched_distance = norm_mean_dist * map_size

    #
    matched_pts = len(matched_pt_pairs) # Green
    missing_pts = np.count_nonzero(true_image) - matched_pts # Pink
    unmatched_pts = np.count_nonzero(pred_image) - matched_pts # Red

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

    # Calculate statistical values
    sum_of_similarities=sum([1-item[1] for item in matched_pt_pairs])
    precision = sum_of_similarities / np.count_nonzero(pred_image)
    recall = sum_of_similarities / np.count_nonzero(true_image)
    f_score = (2 * precision * recall) / (precision + recall)

    return (f_score, precision, recall, mean_matched_distance, matched_pts, missing_pts, unmatched_pts, feedback_image)

def match_nearest_points(true_image, pred_image, min_valid_range=10, parallel_workers=1):
    """
    mat_true, mat_pred: 2d matrices, with 0s and 1s only
    min_valid_range: the maximum distance in % of the largest size of the image (diagonal)
        between a predicted pixel vs. a true one that will be considered
        as valid to include in the scoring.
    calculate_distance: when True this will not only calculate overlapping pixels
        but also the distances between nearesttrue and predicted pixels
    """
    
    lowest_dist_pairs = []
    pred_points_done = set()
    true_points_done = set()

    # perfect prediction pixels
    intersection = pred_image*true_image
    for x, y, _ in np.argwhere(intersection==1):
        lowest_dist_pairs.append((((x, y), (x, y)), 0.0)) 
        true_points_done.add((x, y))
        pred_points_done.add((x, y))
    
    diagonal_length=math.sqrt(math.pow(true_image.shape[0], 2)+ math.pow(true_image.shape[1], 2))
    min_valid_range=int((min_valid_range*diagonal_length)/100) # in pixels

    def nearest_pixels(x, y):
        result=[]
        # find all the points in pred withing min_valid_range rectangle
        mat_pred_inrange=pred_image[
         max(x-min_valid_range, 0): min(x+min_valid_range, true_image.shape[0]),
            max(y-min_valid_range, 0): min(y+min_valid_range, true_image.shape[1])
        ]
        for x_pred_shift, y_pred_shift, _ in np.argwhere(mat_pred_inrange==1):
            y_pred=max(y-min_valid_range, 0)+y_pred_shift
            x_pred=max(x-min_valid_range, 0)+x_pred_shift
            if (x_pred, y_pred) in pred_points_done:
                continue
            # calculate eucledean distances 
            dist_square=math.pow(x-x_pred, 2)+math.pow(y-y_pred, 2)
            result.append((((x, y), (x_pred, y_pred)), dist_square))
        return result

    candidates = [(x, y) for x, y, _ in np.argwhere(true_image==1) if (x, y) not in true_points_done]
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

def usgs_grade_poly_raster(pred_image, true_image, image, legend, feedback_image=None, difficult_weight=0.7):
    intersection = true_image * pred_image
    union = true_image | pred_image
    true_positive = np.count_nonzero(intersection)

    if true_positive == 0:
        return (0, 0, 0, 0, feedback_image)

    if difficult_weight is None:
        precision = true_positive/np.count_nonzero(pred_image)
        recall = true_positive/np.count_nonzero(true_image)

    else:
        hard_pixel_mask = detect_difficult_pixels(image, true_image, lgd_pts=legend['points'])
            
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
    
    if feedback_image is not None:
        feedback_image[(true_image>=1).all(-1)] = MISS_COLOR
        feedback_image[(image>=1).all(-1)] = FAIL_COLOR
        feedback_image[(intersection==1).all(-1)] = CORRECT_COLOR

    f1_score = (2 * precision * recall)/(precision+recall)
    iou_score = true_positive / np.count_nonzero(union)

    return (f1_score, precision, recall, iou_score, feedback_image)

def match_by_color(image, lgd_pts, color_range=20):
    """
    image: the image array for the map image
    lgd_pts: coordinate for the legend feature, from the legend json file
    """
    # get the legend coors and the predominant color
    min_pt, max_pt = boundingBox(lgd_pts)      
    lgd_image = image[min_pt[1]:max_pt[1], min_pt[0]:max_pt[0], :]
    # take the median of the colors to find the predominant color
    median_color = [np.median(lgd_image[:,:,0]), np.median(lgd_image[:,:,1]), np.median(lgd_image[:,:,2])]
    # capture the variations of legend color due to scanning errors
    lower = np.array([x - color_range for x in median_color], dtype="uint8")
    upper = np.array([x + color_range for x in median_color], dtype="uint8")
    # create a mask to only preserve current legend color in the basemap
    mask = cv2.inRange(image, lower, upper)
    detected = cv2.bitwise_and(image, image, mask=mask)
    # convert to grayscale 
    detected_gray = cv2.cvtColor(detected, cv2.COLOR_BGR2GRAY)
    img_bw = cv2.threshold(detected_gray, 127, 255, cv2.THRESH_BINARY)[1]
    # convert the grayscale image to binary image
    pred_by_color = img_bw.astype(float) / 255
    return np.expand_dims(pred_by_color, axis=2)

def detect_difficult_pixels(map_image, true_image, lgd_pts, set_false_as='hard'):
    """
    map_image: the image array for the map image
    true_image: 2D array of any channel (out of 3 present) from the true binary raster image 
    lgd_pts: coordinate for the legend feature, from the legend json file
    set_false_as: when set to 'hard' the pixels that are not within the true polygon area will be considered hard
    """
        
    # detect pixels based on color of legend
    pred_by_color=match_by_color(map_image, lgd_pts, color_range=20)
            
    pred_by_color=(1-pred_by_color).astype(np.uint8) # flip, so the unpredicted become hard pixels
    pred_by_color=true_image*pred_by_color # keep only the part within the true polygon
    
    if set_false_as=='hard':
        # the pixels that are not within the true polygon should are deemed hard pixels
        hard_pixel_mask=(1-true_image) | pred_by_color
    else:
        # the outside pixels will be deemed easy!
        hard_pixel_mask=pred_by_color

    return hard_pixel_mask