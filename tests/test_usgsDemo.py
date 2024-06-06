import os
import usgsDemo
import pandas as pd
from types import SimpleNamespace
from tests.utilities import init_test_log

class Test_usgsDemo:
    def run_test_usgsDemo(self, args, expected_csv_path, log):
        """
        Runs usgsDemo with the given arguments.
        Tests that the csv of results produced by usgsDemo match the expected results.
        This function is not a test itself.
        """
        log.info(f'Running usgsDemo with args: {args}')
        os.makedirs('tests/tmp', exist_ok=True)
        test_csv_path = 'tests/tmp/usgsDemo_results.csv'
        args.output = os.path.dirname(test_csv_path)
        
        # Run usgsDemo to get our test results
        usgsDemo.main(args)
        test_csv = pd.read_csv(test_csv_path).sort_values(by=['Map', 'Feature']).reset_index(drop=True)
        test_csv = test_csv[['Map', 'Feature', 'F1 Score', 'Precision', 'Recall']]
        
        # Load the expected results
        expected_csv = pd.read_csv(expected_csv_path).sort_values(by=['Map', 'Feature']).reset_index(drop=True)
        expected_csv = expected_csv[['Map', 'Feature', 'USGS F1 Score', 'USGS Precision', 'USGS Recall']]
        expected_csv.rename(columns={'USGS F1 Score': 'F1 Score', 'USGS Precision': 'Precision', 'USGS Recall': 'Recall'}, inplace=True)

        # Check that the results are consistent.
        log.info(f'Test CSV:\n{test_csv.to_string()}\nExpected CSV:\n{expected_csv.to_string()}')
        assert (test_csv).equals(expected_csv)
        
        # Clean up
        os.remove(test_csv_path)

    def test_mock_map(self):
        """
        Tests that usgsDemo produces the expected results.
        Data used for this test is a mock map that is 100x100 pixels. The mock map has two point features, 2_pt and 3_pt and two poly features L_poly, TZ_poly.
        """
        log = init_test_log('Test_usgsDemo/test_mock_map')
        # args = '-p tests/data/pred_segmentations -t tests/data/true_segmentations -m tests/data/map_images -l tests/data/legends --min_valid_range 5'
        pred_seg_dir = 'tests/data/pred_segmentations'
        args = SimpleNamespace(
            pred_segmentations=[os.path.join(pred_seg_dir,f) for f in os.listdir(pred_seg_dir) if 'mock_map' in f], 
            true_segmentations='tests/data/true_segmentations', 
            map_images='tests/data/map_images',
            legends='tests/data/legends',
            min_valid_range=5,
            difficult_weight=0.7,
            set_false_as='hard',
            color_range=4)
        expected_csv_path = 'tests/data/mock_map_results.csv'

        self.run_test_usgsDemo(args, expected_csv_path, log)
        log.info('Test passed successfully')

    # def test_MT_OldBaldy_map(self):
    #     """
    #     Tests that usgsDemo produces the expected results.
    #     Data used for this test is the point features from MT_OldBaldyMountain_265833_1989_24000_geo_mosaic, which is from the final evaluation dataset.
    #     This test is commented out beacuse the test data is too large to be commited to the repository.
    #     """
    #     log = init_test_log('Test_usgsDemo/test_MT_OldBaldy_map')
    #     # args = '-p tests/uncommited_data/pred_segmentations/MT_OldBaldyMountain_265833_1989_24000_geo_mosaic* -t tests/uncommited_data/true_segmentations -m tests/uncommited_data/map_images/ -l tests/uncommited_data/legends'
    #     pred_seg_dir = 'tests/uncommited_data/pred_segmentations'
    #     args = SimpleNamespace(
    #         pred_segmentations=[os.path.join(pred_seg_dir,f) for f in os.listdir(pred_seg_dir) if 'MT_OldBaldyMountain' in f], 
    #         true_segmentations='tests/uncommited_data/true_segmentations', 
    #         map_images='tests/uncommited_data/map_images',
    #         legends='tests/uncommited_data/legends',
    #         min_valid_range=0.1,
    #         difficult_weight=0.7,
    #         set_false_as='hard',
    #         color_range=4)
    #     expected_csv_path = 'tests/data/MT_OldBaldy_results.csv'

    #     self.run_test_usgsDemo(args, expected_csv_path, log)
    #     log.info('Test passed successfully')

    # def test_CA_Elsinore_map(self):
    #     """
    #     Tests that usgsDemo produces the expected results.
    #     Data used for this test is the point features from CA_Elsinore, which is from the validation dataset.
    #     This test is commented out beacuse the test data is too large to be commited to the repository.
    #     """
    #     log = init_test_log('Test_usgsDemo/test_CA_Elsinore_map')
    #     # args = '-p tests/uncommited_data/pred_segmentations/CA_Elsinore* -t tests/uncommited_data/true_segmentations -m tests/uncommited_data/map_images/ -l tests/uncommited_data/legends'
    #     pred_seg_dir = 'tests/uncommited_data/pred_segmentations'
    #     args = SimpleNamespace(
    #         pred_segmentations=[os.path.join(pred_seg_dir,f) for f in os.listdir(pred_seg_dir) if 'CA_Elsinore' in f],
    #         true_segmentations='tests/uncommited_data/true_segmentations', 
    #         map_images='tests/uncommited_data/map_images',
    #         legends='tests/uncommited_data/legends',
    #         min_valid_range=0.1,
    #         difficult_weight=0.7,
    #         set_false_as='hard',
    #         color_range=4)
    #     expected_csv_path = 'tests/data/CA_Elsinore_results.csv'

    #     self.run_test_usgsDemo(args, expected_csv_path, log)
    #     log.info('Test passed successfully')