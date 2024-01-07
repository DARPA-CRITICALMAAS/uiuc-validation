import os
import logging
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
from tqdm.contrib.logging import logging_redirect_tqdm

from rasterio.features import sieve 

import src.io as io
import src.utils as utils
from src.grading import gradeRaster

def parse_command_line():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-p','--prediction',
                        required=True,
                        help='The location of the predicted rasters to grade')
    parser.add_argument('-t','--truth',
                        required=True,
                        help='The location of the true segmentations to grade against')
    parser.add_argument('-b','--baseimage',
                        default=None,
                        help='The location of the base color map images')
    parser.add_argument('-o', '--output',
                        default='./',
                        help='The location to write to')
    parser.add_argument('--denoise',
                        action='store_true',
                        help='Flag to enable denoiseing step before grading images.')
    return parser.parse_args()

def main():
    args = parse_command_line()

    # Start logger
    log = utils.start_logger('DARPA_CMAAS_VALIDATION','logs/validationDemo.log', log_level=logging.DEBUG, console_log_level=logging.INFO, writemode='w')

    # Get list of filepaths if directory
    if os.path.isdir(args.prediction):
        prediction_rasters = [os.path.join(args.prediction, f) for f in os.listdir(args.prediction) if f.endswith('tif')]
    else:
        prediction_rasters = [args.prediction]
    
    # Create Output Directory
    if not os.path.exists(args.output) and not os.path.splitext(args.output)[1]:
        os.makedirs(args.output)

    results_df = pd.DataFrame(columns = ['Feature Name','F1 Score', 'IoU Score', 'Recall', 'Precision'])
    with logging_redirect_tqdm():
        pbar = tqdm(prediction_rasters)
        log.info(f'Starting grading of {len(prediction_rasters)} files from {args.prediction}')
        for file in pbar:
            featurename = os.path.basename(os.path.splitext(file)[0])
            log.info(f'Processing {featurename}')
            pbar.set_description(f'Processing {featurename}')
            pbar.refresh()
            
            # Load data
            predict = io.loadGeoTiff(file)
            if os.path.isdir(args.truth) and not os.path.splitext(args.truth)[1]:
                truth = io.loadGeoTiff(os.path.join(args.truth, os.path.basename(file)))
            else:
                truth = io.loadGeoTiff(args.truth)
            
            if predict is None or truth is None:
                log.error(f'Could not grade "{os.path.basename(file)}" as one or more requiried image could not be loaded')
                continue
            img = predict[0]
            true_img, crs, transform = truth

            if args.baseimage:
                if os.path.isdir(args.baseimage) and not os.path.splitext(args.baseimage)[1]:
                    base_img, _, _ = io.loadGeoTiff(os.path.join(args.baseimage, os.path.basename(file)))
                else:
                    base_img, _, _ = io.loadGeoTiff(args.baseimage)
            else:
                base_img = None
        
            # Remove "noise" from image by removing pixel groups below a threshold
            if args.denoise:
                img = sieve(img, 10, connectivity=4)
            
            if args.baseimage is None and args.output != './':
                base_img = np.zeros((*img.shape[:2],3), dtype=np.uint8)

            f1_score, iou_score, recall, precision, debug_img = gradeRaster(img, true_img, base_img)
            results_df.loc[len(results_df)] = {'Feature Name' : featurename, 'F1 Score' : f1_score, 'IoU Score' : iou_score, 'Recall' : recall, 'Precision' : precision}
            
            if debug_img is not None:
                log.info(f'Saving graded image for {featurename}')
                io.saveGeoTiff(os.path.join(args.output, featurename + '.tif'), debug_img, crs, transform)
            log.info(f'Results for "{featurename}" | F1 : {f1_score:.2f}, IOU Score : {iou_score:.2f}, Recall : {recall:.2f}, Precision : {precision:.2f}')

    csv_path = os.path.join(args.output, '#' + os.path.basename(os.path.splitext(args.prediction)[0]) + '_results.csv')
    log.info(f'Finished grading saving results to {csv_path}')    
    results_df.to_csv(csv_path)

if __name__=='__main__':
    main()