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

- **Logistic Regression**
- **Decision Trees**
- **Random Forest**
- **Support Vector Machines (SVM)**
- **Gradient Boosting Machines**

### Evaluation

The models were evaluated using metrics such as:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC-AUC**

## Requirements

To run the project, you need the following Python libraries:

- **pandas**
- **numpy**
- **scikit-learn**
- **matplotlib**
- **seaborn**

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
