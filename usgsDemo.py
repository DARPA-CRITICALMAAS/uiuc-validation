import os
import argparse
import pandas as pd
from usgs_grading_metric import feature_f_score

def parse_command_line():
    """Runs Command line argument parser for pipeline. Exit program on bad arguments. Returns struct of arguments"""
    from typing import List
    def parse_data(path: str) -> List[str]:
        """Command line argument parser for --data. --data should accept a list of file and/or directory paths as an
           input. This function is called on each individual element of that list and checks if the path is valid."""
        # Check if it exists
        if not os.path.exists(path):
            msg = f'Invalid path "{path}" specified : Path does not exist'
            raise argparse.ArgumentTypeError(msg+'\n')
        return path
    
    def post_parse_data(data : List[str]) -> List[str]:
        """Loops over all data arguments and finds all tif files. If the path is a directory expands it to all the valid
           files paths inside the dir. Returns a list of valid files. Raises an argument exception if no valid files were given"""
        data_files = []
        for path in data:
            # Check if its a directory
            if os.path.isdir(path):
                data_files.extend([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.tif')])
            if os.path.isfile(path) and path.endswith('.tif'):
                data_files.append(path)
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
    optional_args.add_argument('-o', '--output',
                        default='results',
                        help='Directory to write the validation feedback to. Defaults to "results"')

    # Flags
    flag_group = parser.add_argument_group('Flags', '')
    flag_group.add_argument('-h', '--help',
                        action='help', 
                        help='show this message and exit')

    args = parser.parse_args()
    args.data = post_parse_data(args.data)
    return args

def main():
    args = parse_command_line()

    # Create output directory if it does not exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    results_df = pd.DataFrame(columns=['Map', 'Feature', 'F1 Score', 'Precision', 'Recall'])

    print(f'Running USGS Grading Metric on {len(args.data)} files')
    potential_map_names = [os.path.splitext(m)[0] for m in os.listdir(args.base_maps) if m.endswith('.tif')]
    for pred_filepath in args.data:
        map_name = [m for m in potential_map_names if m in pred_filepath][0]
        feature_name = os.path.basename(os.path.splitext(pred_filepath)[0])
        map_filepath = os.path.join(args.base_maps, map_name + '.tif')
        true_filepath = os.path.join(args.true_segmentations, os.path.basename(pred_filepath))
        json_filepath = os.path.join(args.legends, map_name + '.json')

        result = feature_f_score(map_filepath, pred_filepath, true_filepath, legend_json_path=json_filepath, min_valid_range=None, difficult_weight=.7, color_range=4, set_false_as='hard')
        results_df.loc[len(results_df)] = {'Map' : map_name, 'Feature' : feature_name, 'F1 Score' : result['f_score'], 'Precision' : result['precision'], 'Recall' : result['recall']}

    csv_path = os.path.join(args.output, '#dataset_results.csv')
    print(f'Finished grading, saving results to {csv_path}')    
    results_df.to_csv(csv_path)

if __name__ == '__main__':
    main()