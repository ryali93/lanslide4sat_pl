# [Landslide4Sense](https://www.iarai.ac.at/landslide4sense/): Multi-sensor landslide detection competition & benchmark dataset

## Contents
- [Landslide4Sense 2022](#landslide4sense-2022)
- [Globally Distributed Landslide Detection](#globally-distributed-landslide-detection)
- [Challenge Description](#challenge-description)
- [Data Description](#data-description)
- [Baseline Code](#baseline-code)
- [Evaluation Metric](#evaluation-metric)
- [Submission Guide](#submission-guide)
- [Awards and Prizes](#awards-and-prizes)
- [Timeline](#timeline)
- [Q&A](#frequently-asked-questions)
- [Citation](#citation)


## Landslide4Sense 2022
![Logo](/image/Competition_figure.png?raw=true "landslide_detection")

The [Landslide4Sense](https://www.iarai.ac.at/landslide4sense/) competition, organized by [Institute of Advanced Research in Artificial Intelligence (IARAI)](https://www.iarai.ac.at/), aims to promote research in large-scale landslide detection from multi-source satellite remote sensing imagery. Landslide4Sense dataset has been derived from diverse landslide-affected areas around the world from 2015 through 2021. This benchmark dataset provides an important resource for remote sensing, computer vision, and machine learning communities to support studies on image classification and landslide detection.
Interested in innovative algorithms for landslide detection using satellite imagery? Join us to help shape Landslide4Sense 2022!

## Globally Distributed Landslide Detection

Landslides are a natural phenomenon with devastating consequences, frequent in many parts of the world. Thousands of small and medium-sized ground movements follow earthquakes or heavy rain falls. Landslides have become more damaging in recent years due to climate change, population growth, and unplanned urbanization in unstable mountain areas. Early landslide detection is critical for quick response and management of the consequences. Accurate detection provides information on the landslide exact location and extent, which is necessary for landslide susceptibility modeling and risk assessment. 
Recent advances in machine learning and computer vision combined with a growing availability of satellite imagery and computational resources have facilitated rapid progress in landslide detection. Landslide4Sense aims to promote research in this direction and challenges participants to detect landslides around the globe using multi-sensor satellite images. The images are collected from diverse geographical regions offering an important resource for remote sensing, computer vision, and machine learning communities. 

## Challenge Description

The aim of the competition is to promote innovative algorithms for automatic landslide detection using remote sensing images around the globe, and to provide objective and fair comparisons among different methods. The competition ranking is based on a quantitative accuracy metric (F1 score) computed with respect to undisclosed test samples. Participants will be given a limited time to submit their landslide detection results after the competition starts. The winners will be selected from the top three solutions in the competition ranking.

Special prizes will be awarded to creative and innovative solutions selected by the competition's scientific committee based on originality, generality, and scalability.

**The competition will consist of two phases:**

**Phase 1 (April 1 - June 14):** Participants are provided with training data (with labels) and additional validation images (without labels) to train and validate their methods. Participants can submit their landslide detection results for the validation set to the competition website to get feedback on the performance (Precision, Recall, and F1 score). The ranking of the submission will be displayed on the online leaderboard. In addition, participants should submit a short description of the methodology (1-2 pages) [here](https://cloud.iarai.ac.at/index.php/s/sYQgdHryGMPQsHa) using the [IJCAI](https://www.ijcai.org/authors_kit) LaTeX styles, and Word templates.

**Phase 2 (June 15 - June 20):** Participants receive the test data set (without labels) and must submit their landslide detection results within 5 days from the release of the test data set. The submissions during that week will be limited to 10 times and only the F1 score will be displayed on the online leaderboard. 

The winners of Phase 2 of the competition will be asked to write a 4-page IJCAI-style formatted manuscript that will be included in the CDCEO workshop. Each manuscript should describe the addressed problem, the proposed method, and the results. The winners will need to prepare a short pre-recorded video presentation for the workshop. The winners should also be present for a live Question-and-Answer period with the audience during the workshop.

The winners **must** submit the working code, the learned parameters, and the manuscript, and present their work in the CDCEO workshop at IJCAI-ECAI 2022 to receive the prizes in accordance with the terms and conditions of the competition.


## Data Description


The Landslide4Sense dataset has three splits, training/validation/test, consisting of 3799, 245, and 800 image patches, respectively. Each image patch is a composite of 14 bands that include:

- **Multispectral data** from [Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2): B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12.

- **Slope data** from [ALOS PALSAR](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-radar-alos-palsar-radar-processing-system): B13.       

- **Digital elevation model (DEM)** from [ALOS PALSAR](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-radar-alos-palsar-radar-processing-system): B14.

All bands in the competition dataset are resized to the resolution of ~10m per pixel. The image patches have the size of 128 x 128 pixels and are labeled pixel-wise.

**Download links:** [training](https://cloud.iarai.ac.at/index.php/s/KrwKngeXN7KjkFm) and [validation](https://cloud.iarai.ac.at/index.php/s/N6TacGsfr5nRNWr).   


![Logo](/image/Data_figure.png?raw=true "landslide_detection")

The _Landslide4Sense_ dataset is structured as follows:
```
├── TrainData/
│   ├── img/     
|   |   ├── image_1.h5
|   |   ├── ...
|   |   ├── image_3799.h5
│   ├── mask/
|   |   ├── mask_1.h5
|   |   ├── ...
|   |   ├── mask_3799.h5
├── ValidData/
|   ├── img/     
|   |   ├── image_1.h5
|   |   ├── ...
|   |   ├── image_245.h5
├── TestData/
    ├── img/     
        ├── image_1.h5
        ├── ...
        ├── image_800.h5
```

Note that the label files (mask files) are only accessible in the training set.

Mapping classes used in the competition:

| Class Number |        Class Name     | Class Code in the Label |
 :-: | :-: | :-:
| 1 | Non-landslide | 0 |
| 2 | Landslide | 1 |


## Baseline Code

This repository provides a simple baseline for the [Landslide4Sense](https://www.iarai.ac.at/landslide4sense/) competition based on the state-of-the-art DL model for semantic segmentation, implemented in PyTorch. It contains a customizable training script for [U-Net](https://arxiv.org/abs/1505.04597) along with the dataloader for reading the training and test samples (see `landslide_dataset.py` in the `dataset` folder).

The provided code can be used to predict baseline results for the competition or as a comparison method for your solutions. Feel free to fork this repository for further use in your work!

**Required packages and libraries:**

- Pytorch 1.10
- CUDA 10.2
- h5py

**To train the baseline model:**

```
python Train.py --data_dir <THE-ROOT-PATH-OF-THE-DATA> \
                --gpu_id 0
```  

Please replace `<THE-ROOT-PATH-OF-THE-DATA>` with the local path where you store the Landslide4Sense data.

The trained model will then be saved in `./exp/`

**To generate prediction maps on the validation set with the trained model:**

```
python Predict.py --data_dir <THE-ROOT-PATH-OF-THE-DATA> \
               --gpu_id 0 \
               --test_list ./dataset/valid.txt \
               --snapshot_dir ./validation_map/ \
               --restore_from ./exp/<THE-SAVED-MODEL-NAME>.pth
```  
Please replace `<THE-SAVED-MODEL-NAME>` with the name of your trained model.

Alternatively, our **pretrained model** is available at    [here](https://cloud.iarai.ac.at/index.php/s/CgbjDRK6B5KYaLE).



The generated prediction maps (in `h5` format) will then be saved in `./validation_map/`

**To generate prediction maps on the test set with the trained model:**

```
python Predict.py --data_dir <THE-ROOT-PATH-OF-THE-DATA> \
               --gpu_id 0 \
               --test_list ./dataset/test.txt \
               --snapshot_dir ./test_map/ \
               --restore_from ./exp/<THE-SAVED-MODEL-NAME>.pth
```  

The generated prediction maps (in `h5` format) will then be saved in `./test_map/`

## Evaluation Metric

The F1 score of the landslide category is adopted as the evaluation metric for the leaderboard:

![](https://latex.codecogs.com/svg.image?F_1=&space;2\cdot&space;\frac{precision\cdot&space;recall}{precision&plus;recall})

With the provided baseline method and the pretrained model, you can achieve the following result on the validation set:

| Validation Set | Precision | Recall | F1 Score |
| :--: | :--: | :--: | :--: |
| U-Net Baseline | 51.75 | 65.50 | 57.82 |

Note that the evaluation ranking is **ONLY** based on the **F1 score of the landslide category** in both validation and test phases.

## Submission Guide

                                                                                            
For both validation and test phases, participants should submit a `ZIP` file containing the prediction files for all test images. Each pixel in the prediction file corresponds to the class category with `1` for *landslide regions* and `0` for *non-landslide regions* (similar to the reference data of the training set).

Specifically, the predictions for each test image should be encoded as a `h5` file with the Byte (uint8) data type, and match the dimensions of the test images (i.e., `128×128`).


The submitted `ZIP` file in the validation phase should be structured as follows:
```
├── submission_name.zip     
    ├── mask_1.h5
    ├── mask_2.h5
    ├── ...
    ├── mask_245.h5
```

The submitted `ZIP` file in the test phase should be structured as follows:
```
├── submission_name.zip     
    ├── mask_1.h5
    ├── mask_2.h5
    ├── ...
    ├── mask_800.h5
```

Sample command for the `ZIP` file generation:
```
cd ./validation_map
zip <THE-SUBMISSION-NAME>.zip ./*
```

## Awards and Prizes
The winners of the competition will be selected from the top three ranked solutions and will be awarded the following prizes:
- First-ranked team: Voucher or cash prize worth **5,000 EUR** to the participant/team and one free IJCAI-ECAI 2022 conference registration

- Second-ranked team: Voucher or cash prize worth **3,000 EUR** to the participant/team and one free IJCAI-ECAI 2022 conference registration

- Third-ranked team: Voucher or cash prize worth **2,000 EUR** to the participant/team and one free IJCAI-ECAI 2022 conference registration

- **Special prizes** will be awarded for creative and innovative solutions selected by the workshop's scientific committee.



## Timeline 


- **April 1 (Phase 1):** Contest opens. Release training and validation data. The validation leaderboard starts to receive submissions.
- **June 12 (Phase 1):** Submit a short description of the methodology (1-2 pages) [here](https://cloud.iarai.ac.at/index.php/s/sYQgdHryGMPQsHa) using the [IJCAI](https://www.ijcai.org/authors_kit) LaTeX styles, and Word templates.
- **June 15 (Phase 2):** Release test data. The validation leaderboard closes and the test leaderboard starts to receive submissions.
- **June 20 (Phase 2):** The test leaderboard stops accepting submissions.
- **June 25:** Winner announcement. Invitations to present at the Special Competition Session at the CDCEO workshop.
- **July 10:** Full manuscript (4-pages, IJCAI formatted) submission deadline and pre-recorded presentation video deadline.

## Frequently Asked Questions
- **Q: What is the rule for registration in the competition?**  
A: The valid participation is determined by the short abstract of the methodology (1-2 pages) where all members of your team should be clearly stated. In other words, only team members explicitly and clearly stated on the short abstract will be considered for the next phase of the competition, i.e. being eligible to be awarded as winners. Furthermore, no overlap among teams is allowed in the test phase of the competition, i.e. one person can only be a member of one team. Adding more team members after submitting the short abstract is not feasible.

- **Q: Is there any limit on the number of submissions?**  
A: In phase 1 (or the validation phase), there is no limit to the number of submissions but participants can have at most 10 results on the leaderboard. Participants can delete their existing submissions and resubmit new results to validate more trials.  
In phase 2 (or the test phase), the number of submissions is strictly limited to 10, and participants will have no permission to delete their existing submissions.

- **Q: Why does my submission raise an error?**  
A: The most possible error for submission happens when participants zip the prediction files with the folder containing these files. To avoid this error, just zip the prediction files directly without a folder inside.

- **Q: What are the 14 bands in the dataset?**  
A: The Landslide4Sense dataset is a composite of 14 bands that include:  
– Multispectral data from Sentinel-2: B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12.  
– Slope data from ALOS PALSAR: B13.  
– Digital elevation model (DEM) from ALOS PALSAR: B14.  
Note that the band 8a in the Sentinel-2 image is omitted in the original data collection, so band 1 to 12 in the dataset corresponds to those 12 bands in Sentinel-2 data.

- **Q: What are the coordinates and dates of acquisition of the data?**  
A: The dataset is collected from different countries and regions all over the world (e.g., Japan and other countries or regions). The detailed geographic information and acquisition time will not be released at the current phase in case participants may directly look for the corresponding high-resolution images to check.

- **Q: Why can’t I get access to the competition forum?**  
A: To access the [competition forum](https://www.iarai.ac.at/landslide4sense/forums/), you need to first log in to the IARAI website.

## Citation
Please cite the following paper if you use the data or the codes: 

```
@article{tbd}
```
