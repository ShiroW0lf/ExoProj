# ExoHunter: Advanced Exoplanet Detection Application
## Comprehensive Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [Data Processing](#data-processing)
4. [Machine Learning Models](#machine-learning-models)
5. [Interface Components](#interface-components)
6. [Core Functions](#core-functions)
7. [Workflows](#workflows)
8. [Technical Details](#technical-details)
9. [Glossary](#glossary)

## Introduction

### Overview
ExoHunter is an advanced application designed to detect exoplanets from stellar light curve data using machine learning techniques. The application provides a user-friendly interface for astronomers, researchers, and citizen scientists to process stellar flux data, train machine learning models, and identify potential exoplanet candidates.

### What are Exoplanets?
Exoplanets are planets orbiting stars outside our solar system. Since direct observation is challenging due to the extreme difference in brightness between stars and planets, indirect detection methods are typically used. The primary method implemented in this application is the transit method, which detects the slight dimming of a star when an orbiting planet passes between the star and the observer.

### Key Features
- Data exploration and preprocessing
- Multiple machine learning model options
- Hyperparameter tuning
- Model evaluation with comprehensive metrics
- Exoplanet hunting on new stellar data
- Interactive visualizations
- Model saving and loading capabilities

## System Architecture

### Overall Structure
ExoHunter is built on a streamlined architecture that separates functionality into distinct modules:

1. **Data Management**: Handles input/output operations, data loading, and caching
2. **Preprocessing**: Performs data cleaning, feature extraction, and transformation
3. **Model Training**: Implements various machine learning algorithms with hyperparameter tuning
4. **Evaluation**: Calculates performance metrics and visualizes results
5. **Exoplanet Detection**: Applies trained models to new data to identify candidates
6. **User Interface**: Provides interactive elements for user input and results display

### Technology Stack
- **Frontend and Backend**: Streamlit (Python-based web application framework)
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, Imbalanced-learn
- **Visualization**: Matplotlib, Seaborn
- **Model Persistence**: Joblib

## Data Processing

### Input Data Format
The application expects CSV files containing stellar flux measurements, with the following format:

1. **Training/Testing Data**:
   - `LABEL` column: Class identifier (1 = Non-Exoplanet, 2 = Exoplanet)
   - `FLUX.1`, `FLUX.2`, etc.: Time series flux measurements

2. **HUNT Data**:
   - `Star Name` column: Identifier for each star
   - `FLUX.1`, `FLUX.2`, etc.: Time series flux measurements

### Preprocessing Pipeline
The data preprocessing involves several steps:

#### 1. Data Loading (`load_data` function)
- Loads CSV files into pandas DataFrames
- Implements caching to improve performance with large datasets

**Technical Details**: Uses `@st.cache_data` decorator to store results in memory after the first run, significantly reducing loading times for subsequent operations.

**Simple Explanation**: Efficiently loads your data files and remembers them to make the app run faster.

#### 2. Data Cleaning and Preparation (`preprocess_data` function)
- Converts label format (1→0, 2→1) to match machine learning conventions
- Separates features (flux values) from target (exoplanet/non-exoplanet labels)
- Checks for missing values and invalid data

**Technical Details**: Returns `X` (feature matrix) and `y` (target vector) suitable for scikit-learn models. Label conversion ensures binary classification format (0/1).

**Simple Explanation**: Prepares your data by organizing it into features (the brightness measurements) and labels (whether it's an exoplanet or not).

#### 3. Feature Extraction (`extract_features` function)
- **Raw Features**: Uses all flux measurements directly
- **PCA**: Reduces dimensionality while preserving variance
- **Statistical Features**: Extracts statistical properties from light curves

**Technical Details**: PCA implementation uses scikit-learn's `PCA` class with configurable components. The function returns either transformed data or both transformed data and the fitted PCA object for later application to test data.

**Simple Explanation**: Helps focus on the most important parts of the data, either by using all measurements or by finding patterns that best explain the variation in star brightness.

### Class Imbalance Handling
- **Random Oversampling**: Duplicates minority class samples
- **SMOTE**: Creates synthetic minority class samples

**Technical Details**: Integrated using the `imblearn` library within scikit-learn pipelines to ensure proper application during cross-validation.

**Simple Explanation**: Balances the dataset when you have many more non-exoplanet stars than exoplanet stars, which helps the model learn both classes equally well.

## Machine Learning Models

### K-Nearest Neighbors (KNN)
**Technical Description**: A non-parametric, instance-based learning algorithm that classifies objects based on the majority class of their k nearest neighbors in the feature space. The model calculates distances between data points and makes predictions based on proximity.

**Key Parameters**:
- `n_neighbors`: Number of neighbors to consider (default: 5)
- `weights`: Weight function ('uniform' or 'distance')

**Strengths**:
- Simple implementation and interpretation
- No training phase (lazy learner)
- Naturally handles multi-class problems
- No assumptions about data distribution

**Limitations**:
- Computationally expensive for large datasets
- Sensitive to irrelevant features and the curse of dimensionality
- Requires feature scaling
- Sensitive to imbalanced datasets

**Simple Explanation**: KNN works like "birds of a feather flock together." It predicts whether a star has an exoplanet by looking at the most similar stars that we already know about. If most similar stars have exoplanets, it predicts this star does too.

### Support Vector Machine (SVM)
**Technical Description**: A discriminative classifier that constructs a hyperplane or set of hyperplanes in high-dimensional space to maximize the margin between classes. SVM can perform non-linear classification by implicitly mapping inputs into high-dimensional feature spaces using kernel functions.

**Key Parameters**:
- `C`: Regularization parameter (controls trade-off between margin maximization and error minimization)
- `kernel`: Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
- `gamma`: Kernel coefficient for 'rbf', 'poly', and 'sigmoid'

**Strengths**:
- Effective in high-dimensional spaces
- Memory efficient as it uses a subset of training points (support vectors)
- Versatile through different kernel functions
- Robust against overfitting in high dimensional spaces

**Limitations**:
- Not directly suitable for large datasets (quadratic scaling with samples)
- Does not provide probability estimates directly
- Sensitive to noise
- Selecting appropriate kernel and parameters can be challenging

**Simple Explanation**: SVM draws a boundary between stars with and without exoplanets. It tries to make this boundary as clear as possible, with the maximum space between the two groups. It can also handle complex, non-linear patterns in the brightness data.

### Random Forest
**Technical Description**: An ensemble learning method that constructs multiple decision trees during training and outputs the class that is the mode of the classes of the individual trees. Each tree is built using a random subset of features and bootstrap samples of the training data.

**Key Parameters**:
- `n_estimators`: Number of trees in the forest
- `max_depth`: Maximum depth of trees
- `min_samples_split`: Minimum samples required to split a node
- `min_samples_leaf`: Minimum samples required at a leaf node

**Strengths**:
- Robust to outliers and noise
- Provides feature importance metrics
- Less prone to overfitting than single decision trees
- Handles high-dimensional data well without feature selection
- Efficient on large datasets

**Limitations**:
- Can be computationally intensive
- More complex and harder to interpret than single decision trees
- May overfit on noisy datasets

**Simple Explanation**: Random Forest builds many decision trees, each looking at different aspects of the star's brightness pattern. Each tree votes on whether the star has an exoplanet, and the majority decision becomes the final prediction. It's like getting opinions from many different experts.

### Ensemble Model (Voting Classifier)
**Technical Description**: A meta-classifier that combines multiple individual classifiers (KNN, SVM, Random Forest) and makes predictions based on aggregated votes. The implementation uses soft voting, where the final prediction is based on the weighted average of class probabilities.

**Key Parameters**:
- `estimators`: List of (name, estimator) tuples
- `voting`: Voting type ('hard' or 'soft')

**Strengths**:
- Often achieves better performance than any single classifier
- Reduces overfitting through model averaging
- Increases stability and robustness
- Compensates for weaknesses in individual models

**Limitations**:
- Increased computational complexity
- Less interpretable than single models
- May not improve performance if base models are highly correlated

**Simple Explanation**: The Ensemble model combines the predictions from multiple different models (KNN, SVM, and Random Forest). It's like getting a second and third opinion before making a diagnosis. This often leads to more accurate predictions than any single model.

## Interface Components

### 1. Introduction Section
**Purpose**: Provides an overview of exoplanets, detection methods, and application functionality.

**Features**:
- Explanatory text
- Sample light curve visualization
- Getting started guidance

**Simple Explanation**: A welcome page that explains what exoplanets are, how they're detected, and how to use the application.

### 2. Data Exploration Section
**Purpose**: Enables users to analyze and visualize their dataset before model training.

**Features**:
- Dataset information display
- Class distribution visualization
- Sample light curve plots
- Data quality checks

**Simple Explanation**: Helps you understand your data by showing statistics and visualizations before you start training models.

### 3. Train Model Section
**Purpose**: Facilitates model selection, configuration, and training.

**Features**:
- Model type selection
- Hyperparameter tuning interface
- Feature processing options
- Class imbalance handling methods
- Cross-validation configuration
- Training progress indicators
- Performance metrics display
- Model saving capability

**Simple Explanation**: Allows you to choose and train different machine learning models with customized settings to detect exoplanets in your data.

### 4. Test Model Section
**Purpose**: Evaluates trained models on separate test datasets.

**Features**:
- Test data upload
- Model loading capability
- Performance metrics calculation
- Confusion matrix visualization
- ROC curve plotting
- Classification report display

**Simple Explanation**: Tests how well your trained model works on new data that it hasn't seen before.

### 5. HUNT Section
**Purpose**: Applies trained models to new stellar data to identify exoplanet candidates.

**Features**:
- Hunt data upload
- Confidence threshold adjustment
- Candidate list generation
- Interactive results table
- Light curve visualization of top candidates
- Potential transit highlighting

**Simple Explanation**: This is where the exciting discovery happens! Upload new star data and let the trained model hunt for potential exoplanets.

### 6. About Section
**Purpose**: Provides background information and resources.

**Features**:
- Project information
- Detection method explanations
- Algorithm descriptions
- External resources links

**Simple Explanation**: Provides background information about the project, how it works, and useful resources for learning more about exoplanets.

## Core Functions

### 1. `train_model`
**Purpose**: Creates and trains a machine learning pipeline with specified parameters.

**Technical Details**:
- Implements train-test splitting with stratification
- Configures feature extraction, scaling, and sampling
- Builds selected model with hyperparameters
- Performs cross-validation
- Fits the complete pipeline to training data
- Returns comprehensive results dictionary

**Parameters**:
- `X`: Feature matrix
- `y`: Target vector
- `model_type`: Model selection ("KNN", "SVM", "RF", "Ensemble")
- `params`: Model hyperparameters dictionary
- `feature_extraction`: Feature processing method ("raw", "pca")
- `cv`: Number of cross-validation folds
- `sampling_strategy`: Class imbalance handling method
- `n_components`: Number of PCA components (if applicable)
- `use_scaler`: Scaling method ("standard", "robust")

**Return Value**: Dictionary containing fitted pipeline, dataset splits, predictions, and metadata

**Simple Explanation**: This function builds and trains your selected model using the settings you've chosen, evaluates how well it performs through cross-validation, and prepares it for exoplanet detection.

### 2. `evaluate_model`
**Purpose**: Calculates and visualizes model performance metrics.

**Technical Details**:
- Computes accuracy, precision, recall, and F1-score
- Generates classification report
- Creates confusion matrix visualization
- Plots ROC curve and calculates AUC
- Visualizes feature importance for tree-based models

**Parameters**:
- `results`: Dictionary containing model pipeline and evaluation data

**Simple Explanation**: Shows you how well your model is performing through various measurements and visualizations, helping you understand its strengths and weaknesses.

### 3. `hunt_exoplanets`
**Purpose**: Applies trained models to new data to identify exoplanet candidates.

**Technical Details**:
- Validates input data format
- Applies the same preprocessing used during training
- Generates predictions and confidence scores
- Filters candidates based on confidence threshold
- Creates interactive visualization of results
- Highlights potential transit features in light curves

**Parameters**:
- `pipeline`: Trained model pipeline
- `scaler`: Fitted scaler object (if separate from pipeline)
- `hunt_data`: DataFrame containing new stellar data
- `expected_flux_count`: Expected number of flux columns
- `confidence_threshold`: Minimum confidence for reporting candidates

**Simple Explanation**: Takes your trained model and applies it to new star data to find the most promising exoplanet candidates, highlighting potential transit signals in the light curves.

### 4. `plot_light_curve`
**Purpose**: Creates visualizations of stellar flux over time with optional transit detection.

**Technical Details**:
- Generates matplotlib figure with flux time series
- Implements simple transit detection algorithm using statistical thresholds
- Highlights potential transit events
- Configures plot aesthetics and labels

**Parameters**:
- `flux_data`: Series or array of flux measurements
- `title`: Plot title
- `detect_transits`: Boolean flag to enable transit detection

**Return Value**: Matplotlib figure object

**Simple Explanation**: Creates a graph showing how a star's brightness changes over time, optionally highlighting dips that might indicate an exoplanet passing in front of the star.

### 5. `data_exploration`
**Purpose**: Analyzes and visualizes dataset characteristics.

**Technical Details**:
- Calculates basic dataset statistics
- Visualizes class distribution
- Plots sample light curves from each class
- Checks for data quality issues
- Summarizes flux statistics

**Parameters**:
- `data`: DataFrame containing the dataset to explore

**Simple Explanation**: Helps you understand your data by showing statistics and example light curves before you start building models.

## Workflows

### 1. Typical Training Workflow
1. **Data Upload**: User uploads a CSV file containing labeled stellar flux data
2. **Data Exploration**: System analyzes and visualizes dataset characteristics
3. **Model Configuration**: User selects model type and configures parameters
4. **Training**: System trains the model with cross-validation
5. **Evaluation**: System displays performance metrics and visualizations
6. **Model Saving**: User saves the trained model for future use

### 2. Testing Workflow
1. **Model Selection**: User either uses previously trained model or uploads saved model
2. **Test Data Upload**: User uploads a separate test dataset
3. **Evaluation**: System applies model to test data and displays metrics
4. **Analysis**: User interprets results to assess model generalization

### 3. Exoplanet Hunting Workflow
1. **Model Selection**: User selects a trained model
2. **Hunt Data Upload**: User uploads new, unlabeled stellar data
3. **Confidence Configuration**: User sets the detection confidence threshold
4. **Detection**: System applies model and identifies candidates
5. **Visualization**: System displays candidate list and light curves
6. **Analysis**: User examines candidates for further investigation

## Technical Details

### Machine Learning Pipeline Architecture
The application implements scikit-learn and imbalanced-learn pipelines to ensure proper order of operations:

```
ImbPipeline([
    ('scaler', StandardScaler()),       # Data scaling
    ('sampler', RandomOverSampler()),   # Class imbalance handling
    ('model', Classifier())             # Selected model
])
```

This ensures that:
1. Scaling is applied before any other operations
2. Sampling is performed only on the training data during cross-validation
3. Model parameters are optimized on the transformed data

### Feature Extraction
For dimensionality reduction with PCA, the implementation preserves the principal components for later application to test and hunt data:

```python
pca = PCA(n_components=n_components)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
```

This maintains consistency in feature transformation across all datasets.

### Class Imbalance Handling
Imbalanced-learn's implementation ensures proper application of sampling techniques:

- **Random Oversampling**: Duplicates existing minority samples
  ```python
  sampler = RandomOverSampler(random_state=42)
  ```

- **SMOTE**: Creates synthetic samples by interpolating between existing minority samples
  ```python
  sampler = SMOTE(random_state=42)
  ```

### Cross-Validation Strategy
Stratified k-fold cross-validation ensures balanced class representation:

```python
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1')
```

This is particularly important for imbalanced datasets to ensure each fold maintains the overall class distribution.

### Model Persistence
Models are saved using joblib for efficient serialization of scikit-learn objects:

```python
joblib.dump(pipeline, filename)
```

This preserves the entire pipeline, including preprocessing steps, ensuring consistent application to new data.

## Glossary

**Exoplanet**: A planet orbiting a star outside our solar system.

**Transit Method**: Detection technique that observes the slight dimming of a star when an orbiting planet passes between the star and the observer.

**Light Curve**: A graph showing the brightness of a star over time.

**Flux**: Measure of the radiant energy emitted by a star, typically used in light curve analysis.

**KNN (K-Nearest Neighbors)**: A machine learning algorithm that classifies objects based on the majority class of their k nearest neighbors.

**SVM (Support Vector Machine)**: A machine learning algorithm that finds an optimal boundary between classes in feature space.

**Random Forest**: An ensemble learning method that combines multiple decision trees to improve prediction accuracy.

**Ensemble Model**: A model that combines multiple base models to improve overall performance.

**PCA (Principal Component Analysis)**: A dimensionality reduction technique that transforms the data into a new coordinate system.

**SMOTE (Synthetic Minority Over-sampling Technique)**: A method to address class imbalance by creating synthetic samples of the minority class.

**ROC Curve (Receiver Operating Characteristic)**: A graphical plot that illustrates the diagnostic ability of a binary classifier as its discrimination threshold is varied.

**AUC (Area Under the Curve)**: A performance metric for binary classification that measures the entire two-dimensional area under the ROC curve.

**Confusion Matrix**: A table used to describe the performance of a classification model by comparing predicted and actual class labels.

**F1 Score**: The harmonic mean of precision and recall, providing a balance between these two metrics.

**Cross-Validation**: A model validation technique that assesses how a model will generalize to an independent dataset.

**Hyperparameter**: A parameter whose value is set before the learning process begins, as opposed to model parameters learned during training.

**Pipeline**: A sequence of data processing components where the output of one component becomes the input to the next.

**Feature Importance**: A measure of how much each feature contributes to the predictions of a model.

**Confidence Threshold**: The minimum probability required to classify a sample as positive (exoplanet).

**Label**: The classification target (exoplanet or non-exoplanet) in supervised learning.
