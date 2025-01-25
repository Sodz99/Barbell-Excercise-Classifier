# Weightlifting-Exercise-Tracker-and-Classifier

-------------------------------------------------
> Barbell Exercise Tracker Setup Guide
-------------------------------------------------


This guide will walk you through the setup process for the Barbell Exercise Tracker project.


-------------------------------------------------
> Prerequisites
-------------------------------------------------

1. Install Python
Download and install Python (preferably version 3.8 or higher) from https://www.python.org/.
Ensure Python is added to your system's PATH during installation.

2. Install Visual Studio Code
Download and install Visual Studio Code from https://code.visualstudio.com/.

3. Open Project Folder
Open the project folder in Visual Studio Code.


-------------------------------------------------
> Required Extensions
-------------------------------------------------

In VS Code, navigate to the Extensions tab and install the following extensions:

Python by Microsoft
Jupyter by Microsoft



-------------------------------------------------
> Configure Settings
-------------------------------------------------

Open Settings in VS Code (User space).
In the search bar, paste the following setting ID:

jupyter.interactiveWindow.textEditor.executeSelection

Enable this setting.

This allows you to run a selection of code directly in the interactive window by selecting the code and pressing Shift + Enter.



-------------------------------------------------
> Setup Conda Environment
-------------------------------------------------


Open the terminal in VS Code.

Create the Conda environment:


   conda env create -f environment.yml


Activate the environment:

  conda activate tracking-barbell-exercises



-------------------------------------------------
> Important File Paths Adjustments
-------------------------------------------------

Before running the scripts, ensure the appropriate relative file paths are correctly set in the following files:
you can find the paths for the below files at the bottom of README.txt file


1. make_dataset.py

at line 9, 11, 18, 27, 101, 211


2. build_features.py

at line 5, 29, 289


3. count_repetitions.py

at line 22


4. remove_outliers.py

at line 12, 235



5. train_model.py

at line 5, 27



6. visualize.py

at line 9, 199




-------------------------------------------------
> Data Information
-------------------------------------------------


The raw data for this project is stored in:

Barbell Exercise Tracker\data\raw\MetaMotion

Data Overview:
The dataset was collected using a Meta Motion Sensor, a wearable device from Ambient Lab, equipped with a 10-axis inertial measurement unit and environmental monitoring sensors (gyroscope, accelerometer, barometric pressure sensor, and ambient light sensor).
Data was gathered during gym workout sessions with five individuals performing various barbell exercises while wearing the device on their wrists.
The raw dataset contains approximately 70,000 entries, each with an epoch timestamp and x, y, and z-values from the sensors.
For analysis, accelerometer and gyroscope data are used to derive motion and orientation insights during exercises.





-------------------------------------------------
> Script Execution Order
-------------------------------------------------



Run the following scripts in the specified order:



1. make_dataset.py 
path - \Barbell Excercise Tracker\src\data\make_dataset.py


2. visualize.py
path - \Barbell Excercise Tracker\src\visualization\visualize.py

3. remove_outliers.py
path - \Barbell Excercise Tracker\src\features\remove_outliers.py

3. build_features.py      
path -   \Barbell Excercise Tracker\src\features\build_features.py

4. DataTransformation.py
path  - \Barbell Excercise Tracker\src\features\DataTransformation.py


5. TemporalAbstraction.py
path - \Barbell Excercise Tracker\src\features\TemporalAbstraction.py

6. FrequencyAbstraction.py
path - \Barbell Excercise Tracker\src\features\FrequencyAbstraction.py


7. train_model.py
path - \Barbell Excercise Tracker\src\models\train_model.py


8. LearningAlgorithms.py
path - \Barbell Excercise Tracker\src\models\LearningAlgorithms.py


9. count_repetitions.py
path - \Barbell Excercise Tracker\src\features\count_repetitions.py



