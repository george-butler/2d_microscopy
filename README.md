# A phenotypic switch in the dispersal strategy of breast cancer cells selected for metastatic colonization

**George Butler<sup>1</sup>, Shirley J. Keeton<sup>1</sup>,Louise J. Johnson<sup>1</sup>, and Philip R. Dash<sup>1</sup>**

<sup><sup>1</sup>School of Biological Sciences, University of Reading, Reading, UK</sup>

	
![quantifying_dispersal_example](/Readme_example_image/quantifying_dispersal.jpg)

This repository is a collection of scripts that can be used to measure the phenotypic traits of migratory cancer cells in 2D phase contrast time-lapse microscopy videos. For more information about the application of these scripts to understand the phenotypic behaviour of experimentally evolved metastatic breast cancer cells then please read our [paper](https://royalsocietypublishing.org/doi/full/10.1098/rspb.2020.2523). Note, these scripts were written to be functional, but not necessarily efficient. As a result, the scripts may need to be adjusted for a resource restricted environment.  



## Workflow
### Cell segmentation and tracking 
The individual cells within each time-lapse video were segmented and then tracked via the [Usiigaci](https://github.com/oist/Usiigaci) cell tracking pipeline. 

### Cell contour extraction 
The contour of each cell can then be extracted for all of the images within a given time-lapse video by using the contour_finder.py script in "/Extract_contours/". The location of directory containing the indiviudal segmented images can be set in line 25. It is important to ensure that the "tracks.csv" file from the Usiigaci tracking output is inside of the "Masks_per_frame" directory as seen in "/Extract_contours/Mask_per_frame". If multiple time-lapse videos have been collected, then lines 64-66 enable each video to be uniquely labeled with a: video number, run number, and video key. The output is then save to a .csv file where the name of the file can be set in line 73. 

### Distance to a neighbour cell
The distance between each cell can then be measured from the output of the contour_finder.py script by using the vision.py script in the "/Measure_distance_to_neighbouring_cells/" directory. The vision.py script will automatically process every .csv file that is present within the "/Measure_distance_to_neighbouring_cells/" directory and save each file with the suffix "_neighbour.csv". As a result, if you have multiple videos in which you want to measure the distance between neighbouring cells, then it is advised to place all of the files into the "/Measure_distance_to_neighbouring_cells/ directory before running. 

Note, the main body of vision.py script is designed to run in parallel across multiple cores within a given system. In its current form the number of available cores will be equal to two less then the total number of cores that are available with in the system. However, this can be adjusted in line 106 by setting "n_jobs" equal to the desired number of cores. 

### Calculating Zernike moments 
Zernike moments can be calculated for every cell in each of the frames by using the output from the contour_finder.py script and the scripts within the "/Zernike_moments_calculations/" directory. 

First, to make the moments invariant to translation and scale the pre_zernike_measure.py script must be run across all of the images collected within a given experiment. The subsquent moments can only be compared if they have all passed through this stage. As a result, if more data is collected, then the old and new images will need to be pushed through the same pipeline. Then, once the script has been run, two metrics will be printed the screen: "avg_radius" and "final_image_size". The "avg_radius" and "final_image_size" values can then be set in lines 125 and 135 of the zernike_preprocessing_wrapper.py script. 

Then, once the "avg_radius" and "final_image_size have been set, the number of moments D can be set in line 122 of the zernike_preprocessing_wrapper.py script. Similarly, the scaling parameter R can be set in line 120 although be advised that a larger scaling parameter will create a greater computational load. Finally, the number of moments and the scaling parameter should also be the same in the zernike_preprocessing_wrapper.py and the pre_zernike_measure.py script. If not, then the pre_zernike_measure.py script should be rerun with the correct values. Furthermore, the number of moments D should also be the same in line 5 of the zernike_postprocessing_wrapper.py script.

Finally, once all of the parameters have been set, the Zernike moments can be calculated by running the wrapper.py script. This script will automatically work through the .csv files that are in the directory and call the correct scripts in sequence. Note, the zernike_preprocessing_wrapper.py script will transform the cell contours to be invariant to translation and to scale. This information, along with the radial Zernike polynomials, is then fed into a C program that calculates the moments themselves, ZernikeMomentsC. Note, the ZernikeMomentsC program is designed to run in Linux and will need to be recompile to run in Windows. Finally, once the moments have been calculated, the outputted text file, "moments.txt", is then reformatted into a pandas dataframe by the zernike_postprocessing_wrapper.py script. 



