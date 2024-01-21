# AI Cure - Team Three Musketeers

This repository contains the code which contains the model designed for the competition AI Cure: Where AI meets health. In this code, we have used a model called Stacking Regressor model that is used to obtain the heart rate of an individual given the parameters.

### Team description
Team name: Three Musketeers
Team members:
1. Akhilesh T S
2. Karthik Sriram S
3. Keerthisree Sai Narne

### Problem statement

Heart rate is a vital physiological parameter reflecting the frequency of cardiac contractions. Influenced by factors like age, fitness, and autonomic nervous system activity, heart rate serves as a key indicator of cardiovascular health. Monitoring heart rate during activities aids in optimizing exercise intensity and assessing overall well-being.The goal is to construct an advanced model capable of accurately predicting an individual's heart rate.

### File Descriptions

The dataset is provided in train_data.csv.

### Implementation Details
- **Data Preprocessing:** The dataset is preprocessed to handle any missing values, perform feature scaling, and encode categorical variables as needed.
- **Train-Test Split:** The dataset is split into a training set and a test set.
- **Model Training:** The Stacking Regressor model is defined along with RidgeCV estimator and is used to train the dataset.
- **Model Evaluation:** The trained model is evaluated on the test data to assess its performance in prediction.

### Requirements
  - Python 3.x   

### Libraries:
  - pandas
  - scikit-learn (sklearn)
  - matplotlib
  - seaborn

### How to Use the Repository
- **Clone the Repository:** Clone this repository to your local machine using the following command:
```
git clone [https://github.com/your_username/repo_name.git](https://github.com/akhilesh1709/aicure_ThreeMusketeers.git)
```
- **Install Dependencies:** Ensure you have Python 3.x installed and the required libraries by running-
```
pip install numpy pandas scikit-learn matplotlib seaborn
```
- **Data Preparation:** Load the dataset or a new dataset in a format suitable for training the model.
- **Run the Code:** Execute the code that trains the model and evaluates its performance on the test data.
- **Analyze Results:** Examine the evaluation metrics and predictions generated by the model to gain insights into its performance.
