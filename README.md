# Bubble detection

This project uses a [Stardist](https://stardist.net/) model, to detect bubbles in the flow boiling experiment.

## Image labels

For labeling we used the [CVAT](https://www.cvat.ai/) image labeler with ellipse labels. The project only works with data that represents elipse labels, so its necessary to use them.

Check out some of the label files in the dataset directories to see how the label files need to be structured.

## Setup and usage

For this project you need a working python installation (I used python version 3.11.7) and pip.

First you should move the data need to your local computer. The currently available data can be found on thelma at: *\\\\Thelma\Thelma\FlowBoilingExperiment\full_data_for_stardist*. Move everything in that directory but the *Outputs* folder to the [process data directory](process_data). If you want to include the already processed data, then copy everything from the *Outputs* folder to the [output directory](output).

Before running any scripts run `pip install -r requirements.txt` to install all the dependencies of the project.

To process the downloaded data you can run the *[detect.py](detect.py)* script, which will output the processed data to the [output directory](output).

Read the script descriptions before running them since they include constants which need to be changed for the script to run properly (file paths and hyper-parameters).

## Updates

**Update 1.1**: 
- *[detect.py](detect.py)* now works with .raww, .tiff, .tif, .png and .jpg image formats and works with variable image size (needs fixed image size only when working with .raww)
- *[train.py](train.py)* now works with variable image sizes with when using .png files (still needs fixed image size when using .raww images)

## Files

Almost all the scripts contain some paramaters which can be changed by changing the constats at the start of the file. The parameters (constants) are described in the script files. 

### Training datasets

If you want to train a model, you first need a dataset. To simplify the process of training you can create a new dataset in the [datasets directory](datasets). Each dataset should contain the following sub-directories:

- *images* - contains the .png images that are used in training.
- *masks* - contains generated masks (the masks are generated by the *[train.py](train.py)* script).
- *rawws* - contains the .raww images that are automatically comverted to the .png file format and saved to the *images* folder by the *[train.py](train.py)* script.
- *csv_data* - contains the .csv label data that is used in training
- *xml_cvat* - contains the .xml labal data from the CVAT labeler (needs to be manually converted to .csv data with *[cvat_to_csv.py](dataset_scripts/cvat_to_csv.py)*)
- *text_data* - contains the .txt label data from our own labeler (needs to be manually converted to .csv data with *[text_to_csv.py](dataset_scripts/text_to_csv.py)*)

*xml_cvat* and *text_data* sub-directories are not necessary since the data can be converted to .csv before inserting it into a dataset. They are only present to make the conversion more convenient with the aforementioned conversion scripts.

You can create the sub-directories manually or you can use the *[create_dataset.py](dataset_scripts/create_dataset.py)* script which automatically generates all the sub-directories in a selected directory

In the current [datasets directory](datasets) you can find a [full dataset](datasets/full_dataset/) that contains all images that were used for training and [small bubbles dataset](datasets/small_bubbles/) which only has small bubbles labeled. You can also find some validation datasets that we used to compare detected and annotated distributions. Their name structures are as follows month_direction_temparature_flow_part (e.g. 03_vertical_50_150_p3)

### Dataset scripts

#### <span name="create_dataset"> [create_dataset.py](dataset_scripts/create_dataset.py) </span>

This script is used to create the sub-directories in a selected dataset directory. It creates *images*, *masks*, *rawws*, *csv_data*, *xml_cvat*, *text_data* sub_directories. 

To use it you should create a folder in the [datasets directory](datasets), then select it when the script prompts you to choose a dataset folder. The selected folder needs to be empty otherwise the script will do nothing.

#### <span name="cvat_to_csv"> [cvat_to_csv.py](dataset_scripts/cvat_to_csv.py) </span>

This script first prompts you to select a dataset directory, then converts .xml files from the *xml_cvat* sub-directory to .csv files in the *csv_data* sub-directory, so they can be used for training.

The *[csv_to_cvat.py](dataset_scripts/csv_to_cvat.py)* script does works the same way but converts files from .csv to .xml.

#### <span name="text_to_csv"> [text_to_csv.py](dataset_scripts/text_to_csv.py) </span>

This script first prompts you to select a dataset directory, then converts .txt files from the *text_data* sub-directory to .csv files in the *csv_data* sub-directory, so they can be used for training.

The *[csv_to_text.py](dataset_scripts/csv_to_text.py)* script does works the same way but converts files from .csv to .txt.

### Model scripts

#### <span name="train"> [train.py](train.py)</span>

This script is used to train a stardist model on a selected dataset. Before we use this script we need to create a dataset which can be created by the *[create_dataset.py](dataset_scripts/create_dataset.py)* script. Then we need to move images and label files to their corresponding folder in the dataset directory, where they are used for training. To set the dataset directory you should change the *DATASET_DIR* constant in the script.

For images you can use .png images or .raww images (I recommend .png for easier use, since .raww images get converted to .png anyways). For labels you can use .csv, .xml or .txt formats (I recommend using .csv since all other formats get converted to .csv by this script anyways. You can also convert between formats by running the conversion scripts described above).

In the script you should change the *MODEL_BASE_DIR* constant to the base directory of the model that you want to train. To change the training settings you can adapt the models *config.json* file in its stardist directory. I recommend using a base learning rate in the [0.0001, 0.001] interval with reduction on plateau. You can find descriptions of other training hyper-parameters in the *Config2D* class in [this file](https://github.com/stardist/stardist/blob/main/stardist/models/model2d.py).

When the model finishes with training, the program optimizes detection and NMS (non-maxima suppression) thresoul, and displays graph of the models performance metrics on the validation data. The last and the best weights of the model are saved to the models stardist directory. You can stop training prematurely with Ctrl + C, this will still optimize thresholds and display performance metrics.

#### <span name="detect"> [detect.py](detect.py)</span>

This script is used to detect bubble labels on new data. You can set the DATA_DIR to set the folder which contains the data, and then set the MERITVE constant, which is the sub-directory containing a set of measurements you want to process. The measurements folder needs to contain sub-directories for each measurement. For bubble detection you NEED to use images in the .raww file format. The structure that the data should follow can be found in the *[process data directory](process_data)*.

You also need to set the output directory by setting the OUTPUT_DIR constant. Detected images (.png), masks (.tiff), labels (.csv) and example images (.png) are written to this folder. The output the aforementioned example data produces can be found in the *[output directory](output)*.

In the directory that stores the set of measurements, you can create a *test_section_bounds.txt* file, which stores a y-coordinate interval that represents the location of test section bounds in the image (two numbers separated by a comma). Any detection outside of this interval is discarded.

### Visualization and analysis scripts

#### <span name="visualizations"> [visualizations.py](visualizations.py)</span>

This script is used to visualize the distributions of a measurement, by displaying the bar graph of volume percentage of bubbles based on their equivalent radius. It prompts you to select a set of measurements (for example you could select *[this directory](output/Example%20measurements%20Vertical/)*) in the *[output directory](output)* and generates graphs for all of the measurements in that directory.

The script also creates a sub-directory called *graphs* in the selected directory where it  stores the generated bar plots.

This script itereates over the folders in steps of 4, so make sure that each measurement condition has 4 measurements/parts (p1, p2, p3, p4).

#### <span name="compare_distributions"> [compare_distributions.py](compare_distributions.py)</span>

This script is used to compare distributions of two measurements. It prompts you to select two measurement folders containing .csv files, display both distributions overlayed on top of each other and the bar plot mentioned above. For the measurement folder you can also select a *csv_data* sub-directory in a dataset (from [this directory](datasets))

#### <span name="compare_labels"> [compare_labels.py](compare_labels.py)</span>

This script is used to visualize the predictions of labels versus the actual annotated labels. First it prompts you to select an image which labels you want to visualize. Then you select the ground truth labels and the predicted labels. 

The script displays an image with both label types overlayed. Truth labels are red and the predictions are displayed as green.

#### <span name="curvature_analysis"> [curvature_analysis.ipynb](curvature_analysis.ipynb)</span>

This script is used for detection of the test section curvature, which helps us analyze the distrotion of the image. It prompts you to select an image that you want to analyze and display parabolas, that outline the test section. It also displays by how many pixels the sides of the image are streched by the camera.

#### <span name="noise_analysis"> [noise_analysis.py](noise_analysis.py)</span>

This script is used to compare the noise in two different images. It prompts you to select two different .png images, then computes the noise level of the image and displays its fourier transform.

#### <span name="source_frequency"> [source_frequency.py](source_frequency.py)</span>

This script is used to detect frequency of bubbles coming from a source. It prompts you to select a folder with .raww images inside and it shows the frequency of bubbles at the selected sources with a fft. Add a .txt file that has the same name as the selected folder to the set of measurements directory in the [pixel data directory](pixel_data), that contains the coordinates of the sources you want to analyze separated with a comma.

### Utils

#### <span name="utils"> [utils.py](utils.py)</span>

This file contains a few functions that are reused throughout the project (mostly for displaying labels on images) and the *RawwReader* class, which is used for reading and processing .raww images.


#### <span name="create_images"> [create_images.py](create_images.py)</span>

This script is used to create images where we conceal bubbles in the top part of the image by replacing it with its background. It prompts you to select an image and its background respectively. Then you can use the keyboard to move the cuts upwards or downwards. When you're done, the image will be saved to the *[representation_images](representation_images)* directory.

The script is used to increase the representation of bubbles at the sides of the test section, which can be underrepresented in the datasets we used.

#### <span name="rotate_raww"> [rotate_raww.py](rotate_raww.py)</span>

This script is used to bulk rotate the .raww images. It prompts you to select a folder containing a set of measurements (for example you could select *[this directory](process_data/Example%20measurements%20Vertical/)*). All images in the measurements in that folder will be rotated by the desired rotation.
