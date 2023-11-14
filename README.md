# [Landslide4Sat PL]: Assessment of Semantic Segmentation Models for Landslide Monitoring Using Satellite Imagery in Peruvian Andes

This repository is associated with the paper "Assessment of Semantic Segmentation Models for Landslide Monitoring Using Satellite Imagery in Peruvian Andes" and includes datasets, models, and scripts utilized in the study.

## Repository Structure
- **data/** `database`: Directory containing the training data used for model development.
- **models/**: Directory where trained models are saved.
- **scripts/**: A collection of scripts for dataset creation, model training, and evaluation.
- **data_validation.xlsx**: Excel file for validation of the collected data.
- **requirements.txt**: A text file listing the dependencies required for the project.
## Scripts Detail
- **config.yaml**: Configuration file for setting up the environment and model parameters.
- **get_data.R** `database`: R script for data retrieval processes.
- **get_data_utils.R** `database`: R utility functions supporting data retrieval.
- **ee_utils.py**: Earth Engine utility functions.
- **dataset.py**: Defines the dataset class used in the models.
- **model.py**: Defines the machine learning models used in the project.
- **train.py**: Script for model training.
- **download_img_gee.py**: Utility script for downloading images.
- **eval_img_gee.py**: Scripts for uploading models to Google Earth Engine.

## Process
### Database generation
1. The [Landslide4Sense dataset](https://www.iarai.ac.at/landslide4sense/challenge/) was downloaded. This database contains 3799 labelled patches in h5 format.
2. The L4S-PE dataset has been expanded with local Peruvian scenarios, totaling 838 annotated images for a comprehensive analysis of landslides in the region. [IRIS](https://github.com/ESA-PhiLab/iris) was used for labelling process.
3. First database was generated with a dimension of 150x150, after labelling process it was splitting to create a 128x128 (wxh)
4. Dataset are separated as L4S (from the challenge) with 3799 images and L4S-PE, the new dataset for peruvian andes with 838 patches.

### Semantic Segmentation Models
- The experiments were configured with [smp](https://github.com/qubvel/segmentation_models.pytorch) using U-net architecture. Additionally, we create a vanilla U-net architecture.
- **Metrics:** Recall, Precision and F1-score.
- **Loss function:** WCE + Dice Loss
- Metrics achieved were:
    - U-net (vanilla) | 14 bands | L4S  : 75.5% 
    - U-net (vanilla) | 14 bands | L4S + L4S+L4Spe: 71.9% 
- All experiments could be find in our [wandb repository](https://wandb.ai/ryali/lanslide4sat_pl?workspace=user-ryali)
 
### Cloud-based Monitoring Framework
We have developed a cloud-based monitoring system leveraging Google Earth Engine, designed to enhance the efficiency and accuracy of landslide detection and monitoring.

## Citation
Please cite the following paper if you use the data or the codes: 

```
@article{tbd}
```