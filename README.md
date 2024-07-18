---

# Heart Disease Prediction using Machine Learning

This repository contains a project for predicting heart disease using various machine learning algorithms. The dataset used is from the UCI Machine Learning Repository and includes several features related to heart health.

## Dataset

The dataset used in this project is the Heart Disease dataset, which consists of 303 instances and 14 attributes:

- `age`
- `sex`
- `cp` (chest pain type)
- `trestbps` (resting blood pressure)
- `chol` (serum cholesterol)
- `fbs` (fasting blood sugar)
- `restecg` (resting electrocardiographic results)
- `thalach` (maximum heart rate achieved)
- `exang` (exercise-induced angina)
- `oldpeak` (ST depression induced by exercise)
- `slope` (slope of the peak exercise ST segment)
- `ca` (number of major vessels)
- `thal`
- `target` (1: presence of heart disease, 0: absence of heart disease)

## Project Structure

The project is structured as follows:

- `heart_data.csv`: The dataset used for training and testing the models.
- `heart_disease_prediction.ipynb`: Jupyter notebook containing the data exploration, model training, and evaluation.

## Getting Started

To get started with this project, you need to have Python and the necessary libraries installed. You can use the following commands to set up your environment:

```bash
pip install numpy pandas scikit-learn
```

## Data Exploration and Preprocessing

The dataset is loaded and basic exploration is performed to understand its structure and check for missing values. Summary statistics and value counts for the `target` variable are also provided.

## Model Training and Evaluation

Three different models are trained and evaluated in this project:

1. **Logistic Regression**
    - Achieved an accuracy of 85.12% on the training data and 81.97% on the test data.
    
2. **Naive Bayes (GaussianNB)**
    - Achieved an accuracy of 81.97% on the test data.

3. **Random Forest Classifier**
    - Achieved an accuracy of 77.05% on the test data.

## Usage

To use the models for prediction, you can input the features for a new instance and get the prediction. An example is provided in the notebook.

```python
input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')
```
