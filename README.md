# Heart Disease Prediction

This project involves analyzing and predicting the risk of heart disease using the Framingham Heart Study dataset. Various machine learning models have been implemented to predict the likelihood of developing coronary heart disease within ten years based on a range of medical and demographic features.

## Dataset

The dataset used in this project is the **Framingham Heart Study dataset**. It contains various health-related metrics and features, including:

- **male**: Gender (binary)
- **age**: Age in years
- **education**: Level of education (integer)
- **currentSmoker**: Whether the person is a current smoker (binary)
- **cigsPerDay**: Number of cigarettes smoked per day
- **BPMeds**: Whether the person is on blood pressure medication (binary)
- **prevalentStroke**: Whether the person has had a stroke (binary)
- **prevalentHyp**: Whether the person has hypertension (binary)
- **diabetes**: Whether the person has diabetes (binary)
- **totChol**: Total cholesterol level
- **sysBP**: Systolic blood pressure
- **diaBP**: Diastolic blood pressure
- **BMI**: Body Mass Index
- **heartRate**: Heart rate
- **glucose**: Glucose level
- **TenYearCHD**: Whether the person developed coronary heart disease

## Implementation

### Models

We implemented several machine learning models for predicting coronary heart disease, including:

- **Logistic Regression**: A regression model used for binary classification that estimates the probability of a binary outcome based on one or more predictor variables.

- **Decision Tree Classifier**: A model that makes decisions based on a tree-like structure of rules. It splits the data into subsets based on feature values.

- **Random Forest Classifier**: An ensemble learning method that combines multiple decision trees to improve classification accuracy and control overfitting.

- **Support Vector Classifier (SVC)**: A model that finds the hyperplane that best separates classes in the feature space. It is effective for high-dimensional spaces and cases where the number of dimensions exceeds the number of samples.

- **Gradient Boosting Classifier**: An ensemble technique that builds models sequentially, with each model correcting the errors of its predecessor, thereby improving performance and reducing bias.

- **AdaBoost Classifier**: An ensemble method that combines multiple weak classifiers to create a strong classifier. It adjusts the weights of misclassified samples to improve accuracy.

- **K-Nearest Neighbors Classifier (KNN)**: A non-parametric method that classifies samples based on the majority class of their nearest neighbors in the feature space.

- **Gaussian Naive Bayes (GaussianNB)**: A probabilistic model based on Bayes' theorem with the assumption of independence among predictors. It is suited for continuous data that follows a Gaussian distribution.

- **XGBoost Classifier**: An advanced gradient boosting technique known for its high performance and efficiency. It includes regularization to prevent overfitting and improve model generalization.

### Evaluation

The models were evaluated using metrics such as:

- **Accuracy**: The ratio of correctly predicted instances to the total instances.
- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives.
- **Recall**: The ratio of correctly predicted positive observations to all observations in the actual class.
- **F1-Score**: The harmonic mean of precision and recall, providing a single metric for model performance.
- **ROC-AUC**: The Area Under the Receiver Operating Characteristic Curve, representing the model's ability to distinguish between classes.

## Requirements

To run the project, you need the following Python libraries:

- **pandas**
- **numpy**
- **scikit-learn**
- **matplotlib**
- **seaborn**
- **xgboost** (for XGBoost Classifier)

## How to Run

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/heart-disease-prediction.git
    ```

2. Navigate to the project directory:

    ```bash
    cd heart-disease-prediction
    ```

3. Install the required libraries:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the main script:

    ```bash
    python main.py
    ```

## Results

The project successfully predicts the likelihood of developing coronary heart disease with a high level of accuracy. The best performing model was the **Random Forest model**, achieving an accuracy of **85%**.

## Contributing

Feel free to contribute to the project by submitting issues or pull requests. Please ensure your code follows the existing style and includes tests where applicable.
