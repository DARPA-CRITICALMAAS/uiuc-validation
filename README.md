## <div align="center">UIUC Validation Metrics</div>
This repository contains the University of Illinois at Urbana-Champaign (UIUC) validation metrics's for DARPA's CriticalMAAS program. This includes the code for generating statistical metrics and debugging feedback on the performance of our models.

## Quickstart

<details>
<summary> Installation </summary>

To get started with this repo you will need to clone the repository and and install [requirements.txt](requirements.txt). We recommend using [**python>=3.10**](https://www.python.org/) and a virtual environment.

```
git clone git@github.com:DARPA-CRITICALMAAS/uiuc-validation.git
cd uiuc-validation
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```
</details>

<details>
<summary> Usage </summary>

### CLI

Included are two demo scripts that can perform generate validation metrics on a file(s) and/or directory(s). 

* **usgsDemo.py** : Will run the orginal USGS grading algorithm on the provided data.
* **validationDemo.py** Will generate a simple F1 score and the USGS F1 Score for the provided data. 

```
# Example call to validationDemo.py
python validationDemo.py --pred_segmentations <mydata>/predictions --true_segmentations <mydata>/true_segmentations --map_images <mydata>/map_images --legends <mydata>/map_legends
```

### Python

There are three provided python methods:

* grade_point_raster : Grades a point raster against the true raster. Will provide the same score as the USGS metric.
* grade_poly_raster : Grades a poly raster against the true raster. The score returned is equivlent to running USGS score with "difficult_weight" = None
* usgs_grade_poly_raster : Grades a poly raster against the true raster. Runs an optimized version of the usgs grading metric for polygons. The score reurned is the same as the USGS metric.

</details>

## Metrics used for grading.

Preforming validation will produce a csv file with scores for each legend feature. The scores provided are:

* F1 Score (equivalent to USGS score with "difficult_weight" = None)
* Precision
* Recall
* IoU Score (Intersection over Union) **Polygon features only*
* USGS F1 Score
* USGS Precision
* USGS Recall
* Mean matched distance (in pixels) **Point features only*
* Matched Points (true positive) **Point features only*
* Unmatched Points (false positive) **Point features only*
* Missing Points (false negative) **Point features only*

## Visual Feedback

If the `--feedback` parameter is enabled the program will also produce a debug image for each legend feature that is graded. Please note that the feedback image is the same size as the orginal image. The key for these images is as follows :

| ${\color{#4F4}\textsf{Correct Prediction (True Positive)}}$ | ${\color{#f00}\textsf{Incorrect Prediction (False Positive)}}$ |
|:-:|:-:|
| ${\color{#000}\textsf{Nothing Present (True Negative)}}$ | ${\color{#f0f}\textsf{Missing Prediction (False Negative)}}$ |

![Example image of feedback for a polygon feature](img/example_poly.png)

<center>
Example of a validation image from a UIUC model run on AR_StJoe_Mbs_poly
</center>

## <div align="center">Documentation</div>
## validationDemo Parameters
* **-p, --pred_segmentations** : required<br>
    Path to file(s) and/or directory(s) containing the predicted rasters to grade. The program will grade any `.tif` files provided. File names are expected to match their corresponding map true raster filename. E.g. if there a file `CA_Sage_Mbv_poly.tif` is provided there needs to be a `CA_Sage_Mbv_poly.tif` file in the true raster directory.
* **-t, --true_segmentations** : required<br>
    Directory containing the true raster segmentations to grade against.
* **-m, --map_images** : required<br>
    Directory containing the base map for the segmentation.
* **-l, --legends** : required<br>
    Directory containing the legend jsons for the maps.
* **-o, --output** : optional<br>
    Directory to write the validation feedback to. Default is "results". The outputs currently created include a visualization image of the validation for each legend and a csv containing the scores for each legend processed. If the directory does not exist, it will be created.
* **--log** : optional<br>
    Option to set the file that logging will write to. Default is "logs/Latest.log".
* **--min_valid_range** : optional<br>
    Maximum distance in % of the largest size of the image (diagonal) between a predicted pixel vs. a true one that will be considered as valid to include in the scoring. Default is 0.1
* **--difficult_weight** : optional<br>
    Weight to give difficult points in the F1 score, range is a float within [0, 1]. Default is 0.7
* **--set_false_as** : optional<br>
    when set to "hard" the pixels that are not within the true polygon area will be considered hard. Set how to treat false positives and false negatives Options are "hard" or "easy". Default is "hard"
* **--color_range** : optional<br>
    The range of color variation to consider for the legend color. Default is 4
* **-v, --verbose** : optional<br>
    Flag to change the default logging level of INFO to DEBUG.
* **--feedback** : optional<br>
    Flag to enable the saving of debugging feedback images.

## Authors and acknowledgment
This repo was created by and is maintained by the UIUC team.
