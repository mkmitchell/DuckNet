
## ReadMe DuckNet

*DuckNet* is an open-source, deep learning-based tool for rapid and accurate duck species identification from drone camera images, as described in Loken et al., submitted. The baseline model is trained to identify the following 8 species: American coot (*Fulica americana*), gadwall (*Mareca strepera*), green-winged teal (*Anas carolinensis*), northern pintail (*Anas acuta*), northern shoveler (*Spatula clypeata*), mallard (*Anas platyrhynchos*), redhead (*Aythya americana*), and ring-necked duck (*Aythya collaris*)

## Running *DuckNet* with a graphical user interface 
The user-friendly graphical interface allows us to combine the automated species identification pipeline with manual review of low-confidence predictions by human experts.

**Step 1. Opening *DuckNet*:** open the unzipped folder and double-click on the `main.bat` file within the *DuckNet* folder. This will open the graphical user interface in your default web browser and a Command prompt window in the background.

![p1](https://user-images.githubusercontent.com/79314212/207560959-91c31490-99da-4c5d-a6c7-06dbfa88bb81.png)



**Step 2. Loading drone camera images:** under the `Files` tab, select *Load Input Images* to load images one-by-one from a folder, or select *Load Input Folder* to load all images from the selected folder. If you load 1000+ images, it might take a few minutes to load all images. A list of all loaded images will appear below.

![p2](https://user-images.githubusercontent.com/79314212/207564616-ba584b25-8352-4da6-bff6-0788b438bdfe.png)



**Step 3. Settings:** under the `Settings` tab choose the *Active model* used for processing the images (default is the baseline_model pre-trained on 13 Central European duck species) and see the species (Known classes) that the model can identify; choose the *Confidence threshold*, below which all predictions will be flagged as „unsure” (default 70%); choose if the coordinates of the bounding boxes should be included in the final csv output file (default is NO); save settings or cancel.

![p3](https://user-images.githubusercontent.com/79314212/207561558-48fb015c-20fd-4570-8399-a2dd54022d03.png)



**Step 4. Process the images:** click on `Process All` to process all loaded images. After processing each image, the species identification(s) and their associated confidence level appear in the `Detected Ducks` tab. In the `Flags` tab the following flags are added: white flag – empty image, no ducks detected in the image; black flag – low-confidence identification, confidence level of species identification is below the set threshold (default 70%, change in the `Settings` tab); checkered flag – multiple ducks in an image. 
 
 ![p4](https://user-images.githubusercontent.com/79314212/207562335-3e878419-04e5-4615-b728-d9bd1bf646b6.png)
 
 
 
**Step 5. Review the detections and identifications:** clicking on a species identification or filename will open the original image, showing a bounding box around each detected duck and the corresponding species identification. Hovering over the species label will show the confidence level of the identification. Images can be sorted based on the confidence level of their identifications, allowing users to quickly review all images below the chosen confidence threshold. Identifications can be removed by clicking on the red X in the upper right corner of each bounding box. Bounding boxes can be resized by pulling the lower right corner of the box and can be moved around by the grey square in the middle of the bounding box. For each image a menu bar appears on the top left corner with four icons:
-	Click on the first icon (black triangle) to process again the image.
-	Clicking on the second icon (eye) allows us to show/hide the bounding boxes from an image (e.g. when multiple identifications are overlapping), and change the brightness of the image.
-	The third icon (bounding box) allows us to add new bounding boxes with species identification to the image (from a drop-down menu or by typing species name), when ducks were not detected in the image. 
-	The fourth icon (question mark) is a summary of short keys: shift + drag to move the image, shift + mouse wheel to zoom and shift + double-click to reset the image. 
 
![p5](https://user-images.githubusercontent.com/79314212/207563431-2c1a1d7f-5ed1-4f74-8a37-4cdea95c0ede.png)



**Step 6. Add Metadata (optional):** this metadata information will be included in the beginning of the csv output file, but it can be left empty. Included fields: site name, site location, site responsible, country, latitude, longitude, camera ID, other; save metadata or cancel.
 
  ![p6](https://user-images.githubusercontent.com/79314212/207563848-ba4c8411-aff7-4576-b2f5-68b73f738a36.png)



**Step 7. Download results:** Under the `Download` tab we can download the results as a csv file  (*Download CSV*) or we can download the bounding boxes as json files to be used as training images (*Download Annotations*). Clicking on the download button will prompt a window to choose the location where the files should be saved. The default filename for the csv file is *detected_ducks.csv*, annotations are saved as the image names with json extension.
 
The output csv file contains the following columns: file name, date when image was taken, time when image was taken, flag (no flag/empty/multiple/unsure), multiple (no multiple/multiple), species (full Latin name of the species or species complex), code (four letter species code), confidence level (on a scale from 0-1).

![p7](https://user-images.githubusercontent.com/79314212/207564272-f21186a4-35e9-462a-9d8f-0fe93f6ddd83.png)





## Running *DuckNet* in batch mode (Windows)
**Step 1.** Run Command prompt (cmd) as Administrator

**Step 2.** Navigate to the unzipped folder where the `main.bat` file is saved 
```
cd C:\DuckNet
```

**Step 3.** Specify the input folder where the drone images are saved (all jpg files will be processed from the target folder), and the folder where the output csv file should be saved. Note: paths that include spaces should be placed between “ “ symbols.
```
main.bat 
--input=\\C:\data\images\image.jpg 
--output=\\C:\data\images\image_species_labels.csv
```

**Additional settings:** specify the model name that should be used for processing the images (default is the ’baseline_model’ pre-trained on 13 Central European bat species); set the confidence threshold (default is 70); set if the coordinates of the bounding boxes should be exported in the output file (default is False). If you do not specify these arguments, the default settings will be used. For further information on how to specify these settings, use `--help` function.
The output csv file contains the following columns: file name, date when image was taken, time when image was taken, flag (no flag/empty/multiple/unsure), multiple (no multiple/multiple), species (full Latin name of the species or species complex), code (four letter species code), confidence level (on a scale from 0-1).



## Retraining *DuckNet*
**Step 1.** under the `Files` tab, select *Load Input Images* to load images one-by-one from a folder or select *Load Input Folder* to load all images from the selected folder. A list of all loaded images will appear below.
 
**Step 2.** under the `Files` tab, select *Load Annotations* and select all corresponding json files that contain the correct bounding box locations and species labels.
 
**Step 3.** Click on the `Training` tab and select *Classes of Interest* (duck species that should be identified), *Unknown classes* (duck species where the human identification was not possible e.g. “Duck_unknown”) and *Rejected classes* (not ducks, but humans, birds and other objects).
 
 ![p8](https://user-images.githubusercontent.com/79314212/207565352-4e01ef21-df82-4e80-bc3d-6b511e28ea29.png)
 
 
 
**Step 4.** Select if only the detector should be retrained or both the detector and the classifier. Retraining only the detector to create a site-specific model is useful when applying *DuckNet* on new backgrounds and the performance of the baseline model is not sufficient. Retraining both the detector and classifier should be used when creating a model that can identify a new species that was not included in our original training data. Annotations for all *Classes of interest* should be included to avoid learning the new species but forgetting the other species. 

![p9](https://user-images.githubusercontent.com/79314212/207565620-a8fcf9b1-682e-4a9c-a7c8-0a7543f0dc56.png)


 
**Step 5.** Set the learning rate (default 0.001) and the number of epochs (default 10) and click on `Start training`. Important: this may take up to several hours or days, depending on the number of training images and your computer performance. After the training finished, save the model. To apply the new model, change the *Active model* under the `Settings` tab. 
 
 ![p10](https://user-images.githubusercontent.com/79314212/207565707-214ebcd7-6611-43d6-aa31-8a20c9ba0bde.png)
 
 
 
**It is highly recommended that before ecological inference users test the model performance (both the baseline or the new, retrained) on a subsample of manually identified drone camera images to uncover possible hidden or new biases.**



### Citation
If you are using this package for a publication, please cite our manuscript:

Citation: Loken, Z., Ringelman, K.M., Mini, A. & Mitchell, M. (submitted) UAVs and Deep Neural Networks Effectively Detect and Identify Non-breeding Waterfowl

### License
LICENSE: CC BY-NC-ND 4.0

### Authors
This repository is maintained by Gabi Krivek & Jaap van Schaik & Alexander Gillert.

### Contact us
If you have any questions or problems with the implementation of *DuckNet*, send an email to Gabi Krivek (krivek.g@gmail.com) or Jaap van Schaik (jaapvanschaik@gmail.com).

### Disclaimer
 *DuckNet* is a free, open-source software without any warranty. You are highly recommended to test the ability of the baseline model (or retrained models) to classify images in your drone image dataset and not assume that the here reported accuracy will be found on your images. The authors of this paper are not responsible for any decisions or interpretations that are made using *DuckNet*.


