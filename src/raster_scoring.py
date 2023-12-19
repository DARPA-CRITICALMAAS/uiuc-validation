import numpy as np

# Plot effects
CORRECT_COLOR = (64,255,64) # Green
FAIL_COLOR = (0,0,255) # Red
MISS_COLOR = (255,0,255) # Fuchsia

def gradeRasterPrediction(img, truth_img, debug_img=None):
    # Do we need to convert to binary mask? Currently i don't think so
    intersection = img & truth_img
    union = img | truth_img

    if debug_img is not None:
        debug_img[(truth_img>=1).all(-1)] = MISS_COLOR
        debug_img[(img>=1).all(-1)] = FAIL_COLOR
        debug_img[(intersection==1).all(-1)] = CORRECT_COLOR

    true_positive = np.count_nonzero(intersection)
    false_positive = np.count_nonzero(img) - true_positive
    false_negative = np.count_nonzero(truth_img) - true_positive

    recall = true_positive / (true_positive+false_positive) 
    precision = true_positive / (true_positive + false_negative)

    f1_score = 2 * ((precision * recall)/(precision+recall))
    iou_score = true_positive / np.count_nonzero(union)

    return (f1_score, iou_score, debug_img)