
# Weightlifting Exercise Tracker and Classifier


A machine learning-based project that automates exercise classification, repetition counting, and form analysis using accelerometer and gyroscope data from wearable devices.

![image](https://github.com/user-attachments/assets/1d23a878-8d68-4565-b7cc-c969a7dd55cc)
Fig. 1. Basic Barbell Exercises




## 📌 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Details](#project-details)
  - [Dataset](#dataset)
  - [Methodology](#methodology)
  - [Results](#results)
- [Future Enhancements](#future-enhancements)
- [Author](#author)
- [License](#license)

---

## 📝 Overview

Strength training is essential for fitness, yet tracking free-weight exercises remains underexplored. This project bridges the gap by leveraging **machine learning models** trained on wearable sensor data to replicate the functionality of a personal trainer, including:
- Tracking exercises
- Counting repetitions
- Detecting improper form

## ✨ Features
- **Exercise Recognition**: Classifies exercises (Bench Press, Deadlift, Squat, Overhead Press, and Row) with high accuracy.
- **Repetition Counting**: Tracks repetitions with ~5% error.
- **Form Analysis**: Detects improper movement patterns.
- **Machine Learning Models**: Optimized algorithms, including **Random Forest** and **Decision Tree**, deliver robust results.
- **Advanced Feature Engineering**: Extracts temporal, frequency, and aggregated features for better predictions.

---

## ⚙️ Installation

### Prerequisites
- Python 3.8 or above
- Required Libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`

### Steps to Set Up
1. Clone this repository using the following command:
   ```bash
   git clone https://github.com/Sodz99/Barbell-Excercise-Classifier.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Barbell-Excercise-Classifier
   ```


2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```


---

## 🚀 Usage

### Run the Project
1. **Preprocess the Dataset**:
   ```bash
   python preprocess_data.py
   ```
2. **Train the Models**:
   ```bash
   python train_models.py
   ```
3. **Visualize Results**:
   ```bash
   python plot_results.py
   ```

### Example Outputs
- Model Performance Metrics
- Confusion Matrices
- Repetition Tracking Visualizations

---

## 🔍 Project Details

### Dataset
- **Sensors Used**: Accelerometer and Gyroscope (MbientLab wristband)
- **Data Points**: 69,677 raw entries processed into 4,505 cleaned instances
- **Participants**: 5 individuals performing barbell exercises (Bench Press, Squat, Deadlift, Overhead Press, Row)

![image](https://github.com/user-attachments/assets/f38c11e9-e1b4-4336-8077-0268c238ad2e)
Fig. 2. Accelerometer Data from Exercise

![image](https://github.com/user-attachments/assets/f5c81383-49de-4fec-ac84-4bfda44cac67)
Fig. 3. Medium and Heavy Weight Squats


### Methodology
1. **Data Preprocessing**:
   - Noise reduction using Butterworth low-pass filtering.
   - Dimensionality reduction with PCA.
   - Aggregating time-series data into features.

2. **Feature Engineering**:
   - Temporal: Standard deviation and mean across time windows.
   - Frequency: Fourier Transform-based features.
   - Clustering: K-Means for label refinement.

3. **Model Development**:
   - Tested Models: Random Forest, K-Nearest Neighbors, Decision Tree, and Naive Bayes.
   - Evaluation Metrics: Accuracy, Precision, Recall, F1-Score.
  
![image](https://github.com/user-attachments/assets/356e7d05-392b-465f-9553-c3650aa81b7e)
Fig. 4. Low-pass Filter

![image](https://github.com/user-attachments/assets/d4018d65-abd2-471c-af39-09c397ba0aad)
Fig. 6. Numerical temporal aggregation with window sizes of 2, 4, and 6 seconds

![image](https://github.com/user-attachments/assets/d3fa3a58-9f23-460d-992d-c2f0507e6c89)
Fig. 5. Principal Component Number

![image](https://github.com/user-attachments/assets/7adb2b36-e187-4288-9eb0-14c42446917c)
Fig. 7. Clusters

![image](https://github.com/user-attachments/assets/25e3fdd6-5a15-43a5-b522-17c256b52898)
Fig. 9. Counting deadlift repetitions using the minimum values after applying a lowpass filter


### Results
- **Best Model**: Random Forest (Accuracy: 99.27%, F1-Score: 99.59%)
- **Generalization**: Achieved 98.99% accuracy on unseen participant data.
- **Repetition Counting**: ~5% error rate using scalar magnitude peaks.

![image](https://github.com/user-attachments/assets/ef471b82-4ef8-4ab8-bc03-9420ab7db1ef)
Fig. 10. Model Performances

![image](https://github.com/user-attachments/assets/c425faf6-a51d-4ee8-b52f-5d6f2625db24)
Fig 11. RF Classification Confusion Matrix



---

## 📈 Future Enhancements
- **Expand Dataset**: Increase diversity in participants and exercises.
- **Real-Time Feedback**: Add live feedback mechanisms for exercise form and progress.
- **Cross-Platform Compatibility**: Integrate with mobile and smartwatch applications.

---

## 👤 Author

**Sohan Arun**  
Master’s Student, Computer Science  
Blekinge Institute of Technology, Sweden  
📧 [Sohanoffice46@gmail.com](mailto:Sohanoffice46@gmail.com)

