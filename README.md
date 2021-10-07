# Bearing Condition State Image Classification

<p align="center">
    <img src="/figures/bearing_missed_predictions.png" | width=800/>
</p>

***Figure: These include actual label (A) and the model predictions (P) made by the classifier. These show some of the missed predictions.***

The four condition state classes in the dataset are:
```
(1) Good
(2) Fair
(3) Poor
(4) Severe
```
***Coming soon in November***
:red_circle:\[[Paper](/access/not_ready.png)\] :green_circle:\[[Dataset](https://doi.org/10.7294/16624642.v1)\] :green_circle:\[[Trained models](https://doi.org/10.7294/16628698.v1)\]

The bearing condition state classification dataset can be used for auxiliary structural inspection tasks to aid in the assessment of structural damage, and to auto-populate bridge inspection reports. 

## Results
We were able to recieve an f1-score of over 86.4% with our trained model. However, note that 100% of the predictions were within one condition state of the ground truth labels.  
<p align="center">
    <img src="/figures/bearing_results.png" | width=400/>
</p>

## Example Use-case:

<p align="center">
    <img src="/figures/workflow.png" | width=800/>
</p>
<p align="center">
    <em> <strong>Example workflow of semi-automated inspection reports</strong> </em>
</p>

<p align="center">
    <img src="/figures/isolation.png" | width=800/>
</p>
<p align="center">
    <em><strong>Isolation of structural detail and extraction into a sub-image</strong></em>
</p>

<p align="center">
    <img src="/figures/material_and_corrosion.jpg" | width=600/>
</p>
<p align="center">
    <em><strong>Material and Corrosion Semantic Segmentation</strong></em>
</p>

<p align="center">
    <img src="/figures/table.jpg" | width=600/>
</p>
<p align="center">
    <em><strong>Sample Inspection Report</strong></em>
</p>

## Requirements
The most important environment configurations are the following:
- Pytorch >= 1.4
- Python >= 3.6
- tqdm
- matplotlib
- sklearn
- cv2
- Pillow
- pandas
- shutil
- pip install efficientnet_pytorch

(you will have to download other modules as they present themselves)

## Evaluating the Trained DeeplabV3+ Model
***WORKING ON THIS SECTION*** - ***COMING SOON***

## Visualizing the results from the Trained EfficientNet B3 Model
***WORKING ON THIS SECTION*** - ***COMING SOON***

## Training with the Bearing Condition State Classification dataset

1. Clone the repository
2. Download the :red_circle:[dataset](/access/not_ready.png)
3. Go into the Training folder of your cloned repository
4. Create a DATA folder
5. Copy and paste the Train and Test folders for 300x300 images from the dataset you downloaded into the DATA folder
6. The DATA folder should have a folder called 'Train' and a folder called 'Test'. Inside each of those folders include the image in their respective folders (1-4). 
7. If you have set this up correctly then you are now ready to begin. Configure and run the **effnet_main.py**.

## Training with a custom dataset

1. Determine which EfficientNet you will need for the size of image you are using. For example, we used EfficientNet B3, which accepts image sizes 300x300 pixels. 
2. Adjust the pre-trained EfficientNet in the **effnet_model.py** file if needed.
3. Configure and run the **effnet_main.py**.

## Extracting sub-images from bounding box data
The bearing dataset was created by extracting all the bounded objects in the [COCO-Bridge-2021+ dataset](/access/not_ready.png). These extracted objects were saved as sub-images to be used for image classification. Here we explain how this process worked so that you can do the same for any dataset or re-create our results. 

***WORKING ON THIS SECTION*** - ***COMING SOON***

## Building a Custom Dataset

0. **If you are planning to extend on the bearing dataset, then please read the annotation guidelines provided by the author in the :red_circle: [bearing dataset](/access/not_ready.png) repository.**

1. We suggest that you use jpeg for the RGB image files. Label/Classify images before they are resized. We advised against beginning with images which are already resized. Before resizing the images, if you want to see some statistics on image heights and widths then you can use the **run_histogram.py**. 

2. For how every many classes you will have folders. In our case he had four folders labeled (1), (2), (3), (4) for each of the condition states. 

3. Put the image which corresponds to the class into the folder. In this way you are labeling the images. 

4. Once this is complete then you can sort the dataset into Test and Training for each class using the **run_random_sort.py** file. Now you are ready to begin training. 

## Citations
***Bearing Condition State Dataset:***
```

```

***Bearing Condition State Model:***
```
Bianchi, Eric; Hebdon, Matthew (2021): Trained Model for the Classification of Bearing Condition States. 
University Libraries, Virginia Tech. Software. https://doi.org/10.7294/16628698.v1 
```

***Paper:***
```

```


