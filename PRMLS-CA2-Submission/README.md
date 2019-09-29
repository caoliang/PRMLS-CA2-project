# Pattern Recognition and Machine Learning Systems Continuous Assessment 2


### Files List

1. Folder Stuctures
   └─sources             ---> source code and data files
    ├─data               ---> Image data HDF5 files
    ├─model              ---> Deep learing model files
    │  ├─base_modell     ---> Base Deep learing model files
    │  ├─ensemble_model  ---> Ensemble Deep learning model files
    │  └─final_model     ---> Final Deep learning model files
    ├─post-processing    ---> Post-processing image analysis files
    ├─pre-processing     ---> Pre-processing application and image files
    │  ├─dt_bird         ---> Bird image files
    │  ├─dt_cat          ---> Cat image files
    │  ├─dt_dog          ---> Dog image files
    │  ├─geckodriver     ---> Selenium driver files used to run downloading application
    │  └─unfit_img       ---> Outliner image files to be excluded from dataset 
    └─sampling           ---> Sample preparation, and training and testing data preparation 

2. Training python scripts
   - sources\model\final_model\PRMLS_CA2_training.py
   
2. Testing python scripts
   - sources\model\final_model\PRMLS_CA2_testing.py


### Steps to create conda environment

1. Download Anaconda installer from web site below
   www.anaconda.com/download/

2. Install Anaconda with the installer

3. Open Anaconda command prompt and execute command below

    conda create -n ca2 python=3.6 numpy=1.15.1 opencv=3.4.2 matplotlib=2.2.3 tensorflow=1.13.1 tensorflow-gpu=1.13.1 cudatoolkit=9.0 cudnn=7.1.4 scipy=1.1.0 scikit-learn=0.19.1 pillow=5.1.0 spyder=3.3.2 cython=0.29.2 pathlib=1.0.1 ipython=7.2.0 yaml pandas keras keras-gpu pydot graphviz opencv

4. Execute Anaconda Naviagator
    - From "Application on channels" drop down box, select "ca2" channel
    - Click "Install" under "Spyder" application
    - Click "Launch" to run "Spyder" application
    
### Steps to run training and testing scripts

1. Run training scripts
1-1. Run "Spyder" application
     - From "Application on channels" drop down box, select "ca2" channel
     - Click "Launch" to run "Spyder" application

1-2. Click toolbar folder icon to change the folder location as below
     - sources\model\final_model\

1-3. Double click the script file to load the script
     - PRMLS_CA2_training.py

1-4. Click "Run Files" icon at toolbar or press F5 key to execute the script

1-5. After execution, rename the model weights file 
     "sources\model\final_model\PRMLS_CA2_training.hdf5"
     to "PRMLS_CA2_testing.hdf5"

2. Run training scripts
2-1. Run "Spyder" application
     - From "Application on channels" drop down box, select "ca2" channel
     - Click "Launch" to run "Spyder" application

2-2. Click toolbar folder icon to change the folder location as below
     - sources\model\final_model\

2-3. Double click the script file to load the script
     - PRMLS_CA2_testing.py

2-4. Click "Run Files" icon at toolbar or press F5 key to execute the script



