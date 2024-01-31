import os
import logging
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
from tqdm.contrib.logging import logging_redirect_tqdm

#from rasterio.features import sieve 

import src.io as io
import src.utils as utils
from src.grading import grade_poly_raster, usgs_grade_poly_raster, usgs_grade_pt_raster

LOGGER_NAME = 'DARPA_CMAAS_VALIDATION'
FILE_LOG_LEVEL = logging.DEBUG
STREAM_LOG_LEVEL = logging.INFO

def parse_command_line():
    """Runs Command line argument parser for pipeline. Exit program on bad arguments. Returns struct of arguments"""
    from typing import List
    def parse_data(path: str) -> List[str]:
        """Command line argument parser for --data. --data should accept a list of file and/or directory paths as an
           input. This function is run called on each individual element of that list and checks if the path is valid
           and if the path is a directory expands it to all the valid files paths inside the dir. Returns a list of 
           valid files. This is intended to be used in conjunction with the post_parse_data function"""
        # Check if it exists
        if not os.path.exists(path):
            msg = f'Invalid path "{path}" specified : Path does not exist'
            #log.warning(msg)
            return None
            #raise argparse.ArgumentTypeError(msg+'\n')
        # Check if its a directory
        if os.path.isdir(path):
            data_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.tif')]
            #if len(data_files) == 0:
                #log.warning(f'Invalid path "{path}" specified : Directory does not contain any .tif files')
        if os.path.isfile(path):
            data_files = [path]
        return data_files
    
    def post_parse_data(data : List[List[str]]) -> List[str]:
        """Cleans up the output of parse data from a list of lists to a single list and does validity checks for the 
           data as a whole. Returns a list of valid files. Raises an argument exception if no valid files were given"""
        # Check that there is at least 1 valid map to run on
        data_files = [file for sublist in data if sublist is not None for file in sublist]
        if len(data_files) == 0:
            msg = f'No valid files where given to --data argument. --data should be given a path or paths to file(s) \
                    and/or directory(s) containing the data to perform inference on. program will only run on .tif files'
            raise argparse.ArgumentTypeError(msg)
        return data_files
    
    def parse_directory(path : str) -> str:
        """Command line argument parser for directory path arguments. Raises argument error if the path does not exist
           or if it is not a valid directory. Returns directory path"""
        # Check if it exists
        if not os.path.exists(path):
            msg = f'Invalid path "{path}" specified : Path does not exist\n'
            raise argparse.ArgumentTypeError(msg)
        # Check if its a directory
        if not os.path.isdir(path):
            msg = f'Invalid path "{path}" specified : Path is not a directory\n'
            raise argparse.ArgumentTypeError(msg)
        return path

    def parse_feature(feature_type : str) -> str:
        """Command line argument parser for --feature_type. Case insensitive, accepts point, pt, polygon or poly.
           Raises an argument exception if argument is one of these list. Returns lowercase feature type"""
        # Convert shorthand to proper form
        feature_type = feature_type.lower()
        if feature_type == 'point':
            feature_type = 'pt'
        if feature_type == 'polygon':
            feature_type = 'poly'
        # Check if feature type is valid
        if feature_type not in ['pt','poly']:
            msg = f'Invalid feature type "{feature_type}" specified.\nAvailable feature types are :\n\t* Point\n\t* \
                    Polygon'
            raise argparse.ArgumentTypeError(msg)
        return feature_type
    
    parser = argparse.ArgumentParser(description='', add_help=False)
    required_args = parser.add_argument_group('required arguments', '')
    required_args.add_argument('-d', '--data', 
                        type=parse_data,
                        required=True,
                        nargs='+',
                        help='Path to file(s) and/or directory(s) containing the predicted rasters to grade. The \
                              program will run grade on any .tif files.')            
    required_args.add_argument('-t','--true_segmentations',
                        type=parse_directory,
                        required=True,
                        help='Directory containing the true raster segmentations to score the data against.')
    required_args.add_argument('-b','--base_maps',
                        type=parse_directory,
                        required=True,
                        help='Directory containing the base map images for the segmentations')
    required_args.add_argument('-l', '--legends',
                        type=parse_directory,
                        required=True,
                        default=None,
                        help='Optional directory containing precomputed legend data in USGS json format. If option is \
                              provided, the pipeline will use the precomputed legend data instead of generating its own.')

    # Optional Arguments
    optional_args = parser.add_argument_group('optional arguments', '')
    optional_args.add_argument('-f','--feature_type',
                        type=parse_feature,
                        default='poly',
                        help=f'Type of features that will be graded on, will be used if the feature type can\'t be \
                               detected from the file name. Available features are Point or Polygon') 
    optional_args.add_argument('-o', '--output',
                        default='results',
                        help='Directory to write the validation feedback to. Defaults to "results"')
    optional_args.add_argument('--log',
                        default='logs/Latest.log',
                        help='Option to set the file logging will output to. Defaults to "logs/Latest.log"')
    # Flags
    flag_group = parser.add_argument_group('Flags', '')
    flag_group.add_argument('-h', '--help',
                        action='help', 
                        help='show this message and exit')
    flag_group.add_argument('-v', '--verbose',
                        action='store_true',
                        help='Flag to change the logging level from INFO to DEBUG')
    flag_group.add_argument('--feedback',
                        action='store_true',
                        help='Enable saving debugging feedback images.')
    # flag_group.add_argument('--denoise',
    #                     action='store_true',
    #                     help='Flag to enable denoiseing step before grading images.')

    args = parser.parse_args()
    args.data = post_parse_data(args.data)
    return args

def main():
    args = parse_command_line()

    # Start logger
    if args.verbose:
        global FILE_LOG_LEVEL, STREAM_LOG_LEVEL
        FILE_LOG_LEVEL = logging.DEBUG
        STREAM_LOG_LEVEL = logging.DEBUG
    global log
    log = utils.start_logger(LOGGER_NAME, args.log, log_level=FILE_LOG_LEVEL, console_log_level=STREAM_LOG_LEVEL)

    # Log info statement to console even if in warning only mode
    log.handlers[1].setLevel(logging.INFO)
    log.info(f'Running pipeline on {os.uname()[1]} with following parameters:\n' +
            f'\tFeature type : {args.feature_type}\n' +
            f'\tData         : {args.data}\n' +
            f'\tTrue Data    : {args.true_segmentations}\n' +
            f'\tMaps         : {args.base_maps}\n' +
            f'\tOutput       : {args.output}\n')
    log.handlers[1].setLevel(STREAM_LOG_LEVEL)

    # Create Output Directory
    if not os.path.exists(args.output) and not os.path.splitext(args.output)[1]:
        os.makedirs(args.output)

    results_df = pd.DataFrame(columns = ['Feature Name', 'F1 Score', 'Weighted F1 Score (polys)', 'Precision', 
                                         'Weighted Precision (polys)', 'Recall', 'Weighted Recall (polys)', 'IoU Score (polys)', 
                                         'Mean matched distance (points)', 'Matched (points)', 'Missing (points)', 
                                         'Unmatched (points)'])
    
    with logging_redirect_tqdm():
        pbar = tqdm(args.data)
        log.info(f'Starting grading of {len(args.data)} files')
        last_map_filepath = None
        last_legend_filepath = None
        potential_map_names = [os.path.splitext(m)[0] for m in os.listdir(args.base_maps) if m.endswith('.tif')]
        for pred_filepath in pbar:
            feature_name = os.path.basename(os.path.splitext(pred_filepath)[0])
            feature_type = feature_name.split('_')[-1]
            if feature_type not in ['pt','poly']: # If feature type can't be figured out automaticallly use default
                feature_type = args.feature_type
            log.info(f'Processing {feature_name}')
            pbar.set_description(f'Processing {feature_name}')
            pbar.refresh()
            
            # Load data
            pred_image, _, _ = io.loadGeoTiff(pred_filepath)
            try:
                true_image, _, _ = io.loadGeoTiff(os.path.join(args.true_segmentations, os.path.basename(pred_filepath)))
            except:
                log.error(f'No true segementation map present for {feature_name}. Skipping')
                continue
            map_name = [m for m in potential_map_names if m in pred_filepath][0]
            map_filepath = os.path.join(args.base_maps, map_name + '.tif')
            if map_filepath != last_map_filepath:
                map_image, crs, transform = io.loadGeoTiff(map_filepath)
                last_map_filepath = map_filepath
            legend_filepath = os.path.join(args.legends, map_name + '.json')
            if legend_filepath != last_legend_filepath:
                map_legend = io.loadLegendJson(legend_filepath)
                last_legend_filepath = legend_filepath
            legend = [s for s in map_legend['shapes'] if s['label'] in pred_filepath][0]

            feedback_image = None
            if args.feedback:
                feedback_image = np.zeros(map_image.shape, dtype=np.uint8)
                
            # Remove "noise" from image by removing pixel groups below a threshold
            #if args.denoise:
            #    img = sieve(img, 10, connectivity=4)

            if feature_type == 'pt':
                # f_score, precision, recall, matched_pts, mean_matched_distance, missing_pts, unmatched_pts, feedback_image
                pt_results = usgs_grade_pt_raster(pred_image, true_image, feedback_image=feedback_image)
                log.info(f'Results for "{feature_name}" | F1 : {pt_results[0]:.2f}, ' + 
                                                        f'Precision : {pt_results[1]:.2f}, ' +
                                                        f'Recall : {pt_results[2]:.2f}, ' +
                                                        f'Mean matched distance : {pt_results[3]:.2f}, ' +
                                                        f'Matched pts : {pt_results[4]}, ' +
                                                        f'Missing pts : {pt_results[5]}, ' +
                                                        f'Unmatched pts : {pt_results[6]}'
                                                        )
                results_df.loc[len(results_df)] = {'Feature Name' : feature_name, 
                                                    'F1 Score' : pt_results[0],
                                                    'Precision' : pt_results[1],
                                                    'Recall' : pt_results[2],
                                                    'Mean matched distance (points)' : pt_results[3],
                                                    'Matched (points)' : pt_results[4],
                                                    'Missing (points)' : pt_results[5],
                                                    'Unmatched (points)' : pt_results[6],
                                                    'Weighted F1 Score (polys)' : np.nan,
                                                    'Weighted Precision (polys)' : np.nan,
                                                    'Weighted Recall (polys)' : np.nan,
                                                    'IoU Score (polys)' : np.nan
                                                    }
                
            if feature_type == 'poly':
                # f1_score, precision, recall, iou_score, feedback_image
                results = grade_poly_raster(pred_image, true_image, feedback_image=feedback_image)
                
                # w_f1_score, w_precision, w_recall, iou_score, feedback_image
                weighted_results = usgs_grade_poly_raster(pred_image, true_image, map_image, legend, feedback_image=feedback_image, difficult_weight=0.7)
                log.info(f'Results for "{feature_name}" | F1 : {results[0]:.2f}, ' + 
                                                        f'Weighted F1 : {weighted_results[0]:.2f}, ' +
                                                        f'Precision : {results[1]:.2f}, ' +
                                                        f'Weighted Precision : {weighted_results[1]:.2f}, ' +
                                                        f'Recall : {results[2]:.2f}, ' +
                                                        f'Weighted Recall : {weighted_results[2]:.2f}, ' +
                                                        f'IoU Score : {results[3]:.2f}')
                results_df.loc[len(results_df)] = {'Feature Name' : feature_name, 
                                                    'F1 Score' : results[0],
                                                    'Precision' : results[1],
                                                    'Recall' : results[2],
                                                    'IoU Score (polys)' : results[3],
                                                    'Weighted F1 Score (polys)' : weighted_results[0],
                                                    'Weighted Precision (polys)' : weighted_results[1],
                                                    'Weighted Recall (polys)' : weighted_results[2],
                                                    'Mean matched distance (points)' : np.nan,
                                                    'Matched (points)' : np.nan,
                                                    'Missing (points)' : np.nan,
                                                    'Unmatched (points)' : np.nan,
                                                    }

            if feedback_image is not None:
                log.info(f'Saving feedback image for {feature_name}')
                io.saveGeoTiff(os.path.join(args.output, 'val_' + feature_name + '.tif'), feedback_image, crs, transform)
            
    csv_path = os.path.join(args.output, '#dataset_results.csv')
    log.info(f'Finished grading, saving results to {csv_path}')    
    results_df.to_csv(csv_path)

if __name__=='__main__':
    main()