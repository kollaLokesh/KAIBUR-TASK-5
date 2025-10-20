# Task 5: Consumer Complaint Text Classification

## Overview
This project performs text classification on **real consumer complaint data** from the Consumer Financial Protection Bureau (CFPB) database. The dataset contains actual consumer complaints and classifies them into 4 categories:
- **0**: Credit reporting, credit repair services, or other personal consumer reports
- **1**: Debt collection  
- **2**: Consumer Loan
- **3**: Mortgage

## Dataset
The project uses the **real Consumer Complaint Database** from `D:\csv\complaints.csv` (6.9GB dataset with millions of real consumer complaints). The script processes this large dataset efficiently by:
- Loading data in chunks to handle memory constraints
- Filtering for the 4 main complaint categories
- Sampling up to 100,000 records for efficient processing
- Preprocessing real complaint narratives

## Project Structure
```
task5-data-science/
├── consumer_complaint_classification.py  # Main analysis script
├── requirements.txt                      # Python dependencies
├── README.md                            # This file
├── eda_analysis.png                     # EDA visualizations (generated)
├── category_text_length.png             # Category analysis (generated)
└── model_evaluation.png                 # Model performance (generated)
```

## Installation

1. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

2. **Download NLTK data (automatically handled by the script):**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Usage

### Run Complete Analysis
```bash
python untitled.py
```

### Key Features

1. **Exploratory Data Analysis (EDA)**
   - Dataset overview and statistics
   - Category distribution analysis
   - Text length and word count analysis
   - Category-wise text length distribution

2. **Text Preprocessing**
   - Lowercase conversion
   - Special character removal
   - Tokenization
   - Stopword removal
   - Lemmatization

3. **Feature Engineering**
   - Text length features (character count, word count, sentence count)
   - Average word length
   - Keyword-based features for each category
   - TF-IDF vectorization

4. **Model Selection and Training**
   - **Naive Bayes**: Fast and effective for text classification
   - **Logistic Regression**: Linear model with good interpretability
   - **Random Forest**: Ensemble method with feature importance
   - **SVM**: Support Vector Machine with linear kernel

5. **Model Evaluation**
   - Accuracy comparison
   - Confusion matrix
   - Precision, Recall, F1-score metrics
   - Cross-validation scores
   - Feature importance analysis (for Random Forest)

6. **Prediction**
   - Function to classify new complaints
   - Probability scores for each category

## Methodology

### 1. Data Preprocessing Pipeline
```
Raw Text → Lowercase → Remove Special Chars → Tokenize → Remove Stopwords → Lemmatize
```

### 2. Feature Extraction
- **Text Features**: TF-IDF vectors (max 5000 features, 1-2 grams)
- **Statistical Features**: Text length, word count, sentence count, average word length
- **Domain Features**: Keyword counts for each complaint category

### 3. Model Training
- 80/20 train-test split with stratification
- 5-fold cross-validation for robust evaluation
- Grid search for hyperparameter tuning (optional)

### 4. Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

## Results

The analysis provides:
1. **EDA Visualizations**: Understanding data distribution and patterns
2. **Model Comparison**: Performance comparison across different algorithms
3. **Confusion Matrix**: Detailed classification results
4. **Feature Importance**: Most important features for classification
5. **Classification Report**: Detailed metrics for each category

## Sample Output
```
=== CONSUMER COMPLAINT TEXT CLASSIFICATION ===
Task 5: Kaiburr Assessment

Loading dataset...
Dataset loaded successfully. Shape: (1000, 2)

=== EXPLORATORY DATA ANALYSIS ===
Dataset shape: (1000, 2)
Missing values:
consumer_complaint_narrative    0
category                        0
dtype: int64

Category distribution:
Credit reporting, repair, or other: 250 (25.0%)
Debt collection: 250 (25.0%)
Consumer Loan: 250 (25.0%)
Mortgage: 250 (25.0%)

=== TEXT PREPROCESSING ===
Applying text preprocessing...
Dataset shape after preprocessing: (1000, 3)

=== MODEL TRAINING AND COMPARISON ===
Vectorizing text data...

C:\Users\kolla\AppData\Local\Temp\ipykernel_20424\1494159811.py:75: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
  data = data.groupby('label', group_keys=False).apply(lambda x: x.sample(min(len(x), SAMPLE_SIZE//4), random_state=RANDOM_STATE))

Training LogisticRegression...
LogisticRegression Accuracy: 0.9873
              precision    recall  f1-score   support

         0.0       0.98      0.99      0.98      2500
         1.0       0.99      0.99      0.99      2500
         2.0       0.99      0.99      0.99      2500
         3.0       0.99      0.98      0.99      2500

    accuracy                           0.99     10000
   macro avg       0.99      0.99      0.99     10000
weighted avg       0.99      0.99      0.99     10000


Training MultinomialNB...
MultinomialNB Accuracy: 0.9768
              precision    recall  f1-score   support

         0.0       0.97      0.97      0.97      2500
         1.0       0.98      0.97      0.98      2500
         2.0       0.98      0.98      0.98      2500
         3.0       0.98      0.98      0.98      2500

    accuracy                           0.98     10000
   macro avg       0.98      0.98      0.98     10000
weighted avg       0.98      0.98      0.98     10000


Training LinearSVC...
LinearSVC Accuracy: 0.9908
              precision    recall  f1-score   support

         0.0       0.98      0.99      0.99      2500
         1.0       1.00      0.99      0.99      2500
         2.0       0.99      1.00      1.00      2500
         3.0       0.99      0.99      0.99      2500

    accuracy                           0.99     10000
   macro avg       0.99      0.99      0.99     10000
weighted avg       0.99      0.99      0.99     10000
```

## Technical Details

### Dependencies
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **matplotlib/seaborn**: Data visualization
- **nltk**: Natural language processing
- **xgboost/lightgbm**: Advanced ensemble methods (optional)

### Performance Considerations
- TF-IDF vectorization with max 5000 features for efficiency
- Lemmatization for better word representation
- Stratified sampling to handle class imbalance
- Cross-validation for robust model evaluation

## Author
KOLLA LOKESH
Date: 2025
