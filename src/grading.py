import numpy as np

# Plot effects
CORRECT_COLOR = (64,255,64) # Green
FAIL_COLOR = (255,0,0) # Red
MISS_COLOR = (255,0,255) # Fuchsia

def gradeRaster(image, truth_image, debug_image=None):
    # Do we need to convert to binary mask? Currently i don't think so
    intersection = image & truth_image
    union = image | truth_image

    if debug_image is not None:
        debug_image[(truth_image>=1).all(-1)] = MISS_COLOR
        debug_image[(image>=1).all(-1)] = FAIL_COLOR
        debug_image[(intersection==1).all(-1)] = CORRECT_COLOR

    true_positive = np.count_nonzero(intersection)
    false_positive = np.count_nonzero(image) - true_positive
    false_negative = np.count_nonzero(truth_image) - true_positive

    # avoid div by 0
    if true_positive == 0:
        return (0, 0, 0, 0, debug_image)

    recall = true_positive / (true_positive+false_positive) 
    precision = true_positive / (true_positive + false_negative)

    f1_score = 2 * ((precision * recall)/(precision+recall))
    iou_score = true_positive / np.count_nonzero(union)

    return (f1_score, iou_score, recall, precision, debug_image)