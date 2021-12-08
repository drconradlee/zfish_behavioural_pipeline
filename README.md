# Pipeline for Zebrafish Behaviour (Bipeline)

Complete module + executable for segmenting multiwell videos and tracking using Deep-Lab-Cut. This forms the behavioural pipeline (Bipeline) for the Scott Laboratory at the Queensland Brain Institute. This repository and code combines scripts from https://github.com/drconradlee/zfish_video_segmentation and https://github.com/Scott-Lab-QBI/beh_video_processing to minimize user interference. 

<h2> Installation </h2>
You can install all required packages by creating an environment with all dependencies with the included `environment.yml` file.
<p> </p>

```
conda env create -f environment.yml -n bipeline
```

<p> </p>
 

<h2> Activating the environment </h2>
If you are running the script through the installed envionemnt, simply activate the environment and set the current directory.
<p> </p>

```
conda activate bipeline
cd <directory of installed package>
```

<h2> Script Inputs </h2>
To run, simply input:
<p> </p>

```
python bipeline.py -d <path_to_folder> -n <number of wells>
```
If errors are encounters, other input variable are avaliable to fine tune the segmentation. Several input parameters are avaliable and have been put on default for ease of use. A description of all input variables are avaliable on original documentation at https://github.com/drconradlee/zfish_video_segmentation and https://github.com/Scott-Lab-QBI/beh_video_processing.


<h2> "One-click" scripts</h2>

Batch files are included for single click analysis. Simply place the batch script into the folder of raw data, and double click to intitate entire behavioural pipeline. 1,4,7,9,16 well scripts are included. Feel free to edit variables as descibed above for desired input.
