# EEG Data Processing Pipeline
## Team Members

| Name                | Student ID  | Role                                                                                     |
|---------------------|-------------|-----------------------------------------------------------------------------------------|
| Nguyễn Văn Kinh     | 22280051    | - Create image data from preprocessed data                                              |
|                     |             | - EDA on the image data                                                                 |
|                     |             | - CNN for EEG data                                                                      |
| Vũ Đăng Khôi        | 22280049    | - Leader                                                                               |
|                     |             | - Data preprocessing                                                                   |
|                     |             | - Multi-Layer Perceptron for preprocessed data, added features (including CWT)          |
| Mạc Minh Phúc        | 22280065    | - Data preprocessing                                                                   |
|                     |             | - Model XGBoost for preprocessed data, added features (including CWT)                  |
| Nguyễn Lê Lâm Phúc  | 22280066    | - Feature extraction based on Fourier and Continuous Wavelet Transform (CWT)           |
|                     |             | - ML models for added features data                                                    |


## 1. Introduction
EEG, or electroencephalography, is a technique used to record electrical activity from the brain. It typically utilizes a system with 1 to 250 electrodes to measure signals, which represent the collective activity of large groups of neurons working together rather than the activity of individual neurons.

After processing the raw EEG data, we obtain a dataset consisting of **304,540 rows and 14 columns**. These 14 columns correspond to the EEG channels: `AF3`, `F7`, `F3`, `FC5`, `T7`, `P7`, `O1`, `O2`, `P8`, `T8`, `FC6`, `F4`, `F8`, and `AF4`.

---

## 2. Data Preprocessing

### 2.1 Insight Knowledge

#### 2.1.1 Types of Noise in Brain Waves
- **Stereotyped Artifacts**: Predictable noise patterns caused by external interference or regular biological processes, such as:
  - Electrical line noise (50/60 Hz).
  - Physiological activities like eye movements, blinking, or muscle activity.  
  Their consistent and systematic nature makes them easier to identify and mitigate.

- **Non-Stereotyped Artifacts**: Irregular, unpredictable noise patterns caused by transient, non-repeating events, such as:
  - Head movements.
  - Electrode shifts.
  - Sudden muscle contractions.
  - Environmental disturbances.  
  These artifacts are harder to address with automated techniques due to their variability in amplitude, duration, and spatial distribution. They can also disrupt advanced signal processing methods like **Independent Component Analysis (ICA)** because their unpredictability violates spatial and temporal consistency assumptions.

#### 2.1.2 High-Frequency and Low-Frequency Waves
- High-frequency data exhibit a stationary trend, while low-frequency data show a non-stationary trend (based on autocorrelation and KL divergence).  
- **ICA** does not perform well with non-stationary trends.

#### 2.1.3 Sensor Positions
Channels located closer to the eyes are more susceptible to noise from blinks and eye movements.

#### 2.1.4 The Bad Channels
- Channels like `AF4`, `AF3`, `FC5`, `T7`, `T8`, `FC6`, `F4`, and `F8` are highly affected by artifacts.  
- **Decision**: Drop channels `AF3` and `AF4` due to their proximity to the eyes and facial muscles, which makes them more susceptible to noise from blinking, eye movements, and muscle contractions.


---

### 2.2 Preprocessing Steps

#### 2.2.1 Filtering
- Remove low frequencies (<1 Hz) as they tend to be non-stationary.
- Remove high frequencies (>40 Hz) to eliminate noise.  
  *Note: Filtering alone is insufficient.*

#### 2.2.2 Segmentation
- Segment the data into 5-second epochs.  
  This reduces signal stationarity and increases the duration of usable data.

#### 2.2.3 Pre-Cleaning for ICA
- Identify bad epochs (e.g., those with high variance on specific channels or non-stereotyped noise).
- Use the **AutoReject library** to create cleaner data for ICA processing.

#### 2.2.4 ICA (Independent Component Analysis)
- Fit ICA on the cleaned data, where high-variance artifacts have been removed.

#### 2.2.5 Threshold for Removing Noisy Components
- Gradually decrease the correlation threshold between channels and eye-related channels (`AF3` and `AF4`) to isolate and remove noisy components.

## 3. Feature Engineering

### 3.1 Using Transform Methods

#### 3.1.1 Using Fourier Transform to Process Data
- **Definition**:  
  Fourier Transform is the transformation of a function or a signal from the time domain to the frequency domain.



- **Applications in EEG Processing**:  
  - Brain waves are oscillating signals with multiple frequency components (Delta, Theta, Alpha, Beta, Gamma), each corresponding to specific brain functions.
  - Brain wave data in the time domain is complex. Fourier Transform converts the signal into the frequency domain, making it easier to extract and analyze key features.

#### 3.1.2 Using Continuous Wavelet Transform (CWT) to Create Features
- **Definition**:  
  The Continuous Wavelet Transform (CWT) analyzes time-domain signals and provides information in both time and frequency domains.


- **Applications in EEG Processing**:  
  CWT provides simultaneous time-frequency information, detects local changes, and captures complex features like amplitude and frequency variations.

#### 3.1.3 Applying Fourier Transform and CWT to the Problem
1. With the processed dataset reduced to nearly **6 million rows**, image data is created by taking **128 continuous samples** with a **0.3 overlap** per person over the last 5 days (4 days for the fifth person).
2. **Fast Fourier Transform (FFT)** is applied to convert the signal. FFT uses a divide-and-conquer approach to reduce computations and reuse intermediate results.
3. Adding features enhances the dataset and improves classification accuracy:
   - **Frequency Range Analysis**: Each state has distinct frequency characteristics; brain waves can be divided into frequency ranges.
   - **Statistical Features**: Calculate mean, variance, and standard deviation for each channel to create new features.
   - **Continuous Wavelet Transform (CWT)**: Generate **8 additional features per channel** using the Morlet wavelet.



---

### 3.2 Create Image Data from Original Data

#### 3.2.1 Feature Selection
- **Theta wave frequency range (4-7 Hz)**: Appears during deep relaxation or drowsiness, but not when fully alert.
- **Alpha wave frequency range (8-13 Hz)**: Appears in a state of light relaxation, not stressed but not asleep.
- **Beta wave frequency range (16-30 Hz)**: Prominent when the brain is highly focused.  



#### 3.2.2 Method for Creating Image Data
1. Use the **Cubic Interpolation Algorithm** to calculate the values of empty pixels based on the average values of the waves at the electrodes.
2. Combine the three images generated for each wave frequency into one composite image.



## 4. Model

### 4.1 XGBoost
- **Handling Data Imbalance**:  
  - Downsample the majority class using `sklearn.utils.resample`.  
  - Upsample the minority classes using **SMOTE (Synthetic Minority Oversampling Technique)**.

---

### 4.2 Convolution Neural Network (CNN)
- **Handling Data Imbalance**:  
  - Use **class weight** for the loss function.  
  - Apply **oversampling** to balance the dataset.

---

### 4.3 Multilayer Perceptron (MLP)
- **Handling Data Imbalance**:
  1. Select a subset of samples from **label 2** of size:  
     \[
     \text{(samples of label 1 + samples of label 2)}/2 \times 1.3
     \]  
     This prevents the model from overly predicting new samples as **label 2**.
  2. Predict all samples of **label 2**, sort the predictions by accuracy, and extract a subset with the lowest accuracy, maintaining the same size:  
     \[
     \text{(samples of label 1 + samples of label 2)}/2 \times 1.3
     \]  
     Repeat the process of training and predicting iteratively.

---

### 4.4 Stack Classifier
- **Handling Data Imbalance**:  
  - Downsample the **drowsy** and **focused** labels to match the number of **unfocused** labels.

---

### 4.5 Comparison of Models

| **Model/Metric**       | **Preprocessed Data** | **Data with Added Features (incl. CWT)** |
|-------------------------|-----------------------|------------------------------------------|
| **XGBoost**            | Accuracy: 0.71        | Accuracy: 0.89                           |
|                         | F1 Score: 0.71       | F1 Score: 0.89                           |
|                         | AUC Score: 0.88      | AUC Score: 0.96                          |
| **CNN**                | Accuracy: 0.62        |                                          |
|                         | F1 Score: 0.63       |                                          |
|                         | AUC Score: 0.84      |                                          |
| **MLP**                | Accuracy: 0.58        | Accuracy: 0.83                           |
|                         | F1 Score: 0.57       | F1 Score: 0.82                           |
|                         | AUC Score: 0.76      | AUC Score: 0.94                          |
| **Stack Classifier**    |                       | Accuracy: 0.84                           |
|                         |                       | F1 Score: 0.84                           |
|                         |                       | AUC Score: 0.96                          |

- **Insights**:  
  - XGBoost consistently outperforms the other models on both datasets.  
  - MLP has the lowest performance across all evaluation metrics.

---

## 5. Challenges and Improvements

### 5.1 Challenges Faced
1. **Class Imbalance**:  
   Uneven class labels can bias the model towards the majority class, reducing performance for minority classes.
2. **Noise in Data**:  
   EEG signals are often contaminated by noise from muscle movements, eye blinks, and external disruptions, obscuring significant patterns.
3. **Overfitting**:  
   Complex models may overfit, especially with limited training data, leading to poor generalization on unseen data.
4. **Feature Selection**:  
   High-dimensional datasets can include irrelevant or redundant features, negatively impacting model performance.

---

### 5.2 Strategies to Improve Accuracy
1. **Preprocessing**:
   - Employ advanced noise removal techniques, such as **Independent Component Analysis (ICA)**.  
   - Filter low and high frequencies.
2. **Addressing Class Imbalance**:
   - Use oversampling methods like **SMOTE** or undersampling.  
   - Apply **class-weighting** during model training.
3. **Feature Engineering**:
   - Create a brainwave distribution image by extracting features of different wave types.
   - Use **CWT** and statistical attributes to generate additional features.
4. **Hyperparameter Optimization**:
   - Perform grid search to fine-tune model parameters for better performance.

