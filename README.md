# FixaTons
A collection of Human FixationsDatasets and Metrics for Scanpath Similarity

## Citations
If you intend to use this collection of datasets on your research, please cite the technical report

- FixaTons technical report: COMING SOON

and also, all the correspondent pubblications for the dataset included:


- MIT1003: http://people.csail.mit.edu/tjudd/WherePeopleLook/Docs/wherepeoplelook.pdf

 Authors of compatible datasets are encouraged to add them to this collection. Simply send an email to the corresponding author of this report along with an authorization to redistribute that data. Below, the list of datasets currently added to the collection.

    
## Download FixaTons

- Clone or download the folder with the code
- Download the zip file from https://drive.google.com/open?id=1TQSaq5J0p_oCdkyVZ-IzBltLwJ2cm3UA and extract its content in the same folder with the code

## Structure of FixaTons

________________________________________________________________________________

- FixaTons

    - DATASET_NAME

        - STIMULI : contains original images.
                  They can have different file format (jpg, jpeg, png,...)

        - SCANPATHS : contains one folder for each image

            - IMAGE_ID :
                  it contains one file for each scanpath of that image
                  scanpaths are matrices
                  rows of this matrices describe fixations
                  each fixation is of the form :
                  [x-pixel, y-pixel, initial time, final time]
                  Times are in seconds.

        - FIXATION_MAPS : contains a fixation map of each original image
            they are matrices of zeros (NON-fixated pixels) and 255's (fixated
            pixels). They can have different file format (jpg, jpeg, png,...)

        - SALIENCY_MAPS : contains saliency maps of each original image
            they are generated from human data. They can have different file
            format (jpg, jpeg, png,...)
            
________________________________________________________________________________

## How to use FixaTons (with python)


For an easy use of the dataset, python software is provided. This tools help you in different tasks:

    - List information
    - Get data (matrices)
    - Visualize data
    - Compute metrics
    - Compute statistics

For some example codes, please refer to the correspondent report or Tutorial.ipynb (jupiter notebook) https://github.com/dariozanca/FixaTons/blob/master/Tutorial.ipynb.

