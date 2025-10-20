"""
Consumer Complaint Text Classification
Task 5: Kaiburr Assessment

This script performs text classification on consumer complaint dataset
using real data from D:\csv\complaints.csv

Categories based on Product field:
- Credit reporting, credit repair services, or other personal consumer reports
- Debt collection
- Consumer Loan
- Mortgage
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class ConsumerComplaintClassifier:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.label_encoder = LabelEncoder()
        self.vectorizer = None
        self.model = None
        self.category_mapping = {
            0: 'Credit reporting, repair, or other',
            1: 'Debt collection', 
            2: 'Consumer Loan',
            3: 'Mortgage'
        }
    
    def load_data(self, file_path):
        """Load and preprocess the consumer complaint dataset"""
        print("Loading real complaints dataset...")
        
        if not file_path or not file_path.endswith('.csv'):
            raise ValueError("Please provide a valid CSV file path")
        
        print(f"Loading CSV file: {file_path}")
        
        # Read in chunks due to large file size
        chunk_list = []
        chunk_size = 100000  # Read 100k rows at a time
        
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            chunk_list.append(chunk)
            if len(chunk_list) >= 10:  # Limit to 1M rows for processing
                break
        
        self.df = pd.concat(chunk_list, ignore_index=True)
        print(f"Loaded {len(self.df)} rows from dataset")
        
        # Preprocess the real dataset
        self.df = self.preprocess_real_data()
        
        print(f"Dataset loaded successfully. Shape: {self.df.shape}")
        return self.df
    
    def preprocess_real_data(self):
        """Preprocess the real complaints dataset"""
        print("Preprocessing real dataset...")
        
        # Filter for the main categories we want to classify
        target_categories = [
            'Credit reporting, credit repair services, or other personal consumer reports',
            'Debt collection',
            'Consumer Loan',
            'Mortgage'
        ]
        
        # Filter data for our target categories
        self.df = self.df[self.df['Product'].isin(target_categories)]
        
        # Remove rows with missing complaint narratives
        self.df = self.df.dropna(subset=['Consumer complaint narrative'])
        
        # Create category mapping
        category_mapping = {
            'Credit reporting, credit repair services, or other personal consumer reports': 0,
            'Debt collection': 1,
            'Consumer Loan': 2,
            'Mortgage': 3
        }
        
        # Map product categories to numerical labels
        self.df['category'] = self.df['Product'].map(category_mapping)
        
        # Rename columns for consistency
        self.df = self.df.rename(columns={'Consumer complaint narrative': 'consumer_complaint_narrative'})
        
        # Keep only necessary columns
        self.df = self.df[['consumer_complaint_narrative', 'category']].copy()
        
        # Remove any remaining NaN values
        self.df = self.df.dropna()
        
        # Sample data if too large (for faster processing)
        if len(self.df) > 100000:
            print(f"Sampling 100,000 rows from {len(self.df)} total rows")
            self.df = self.df.sample(n=100000, random_state=42)
        
        print(f"Preprocessed dataset shape: {self.df.shape}")
        print(f"Category distribution:\n{self.df['category'].value_counts()}")
        
        return self.df
    
    def exploratory_data_analysis(self):
        """Perform exploratory data analysis"""
        print("\n=== EXPLORATORY DATA ANALYSIS ===")
        
        # Basic statistics
        print(f"Dataset shape: {self.df.shape}")
        print(f"Missing values:\n{self.df.isnull().sum()}")
        
        # Category distribution
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        category_counts = self.df['category'].value_counts()
        plt.pie(category_counts.values, labels=[self.category_mapping[i] for i in category_counts.index], 
                autopct='%1.1f%%', startangle=90)
        plt.title('Distribution of Complaint Categories')
        
        plt.subplot(2, 2, 2)
        category_counts.plot(kind='bar', color='skyblue')
        plt.title('Complaint Categories Count')
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Text length analysis
        plt.subplot(2, 2, 3)
        text_lengths = self.df['consumer_complaint_narrative'].str.len()
        plt.hist(text_lengths, bins=30, color='lightgreen', alpha=0.7)
        plt.title('Distribution of Text Lengths')
        plt.xlabel('Character Count')
        plt.ylabel('Frequency')
        
        # Word count analysis
        plt.subplot(2, 2, 4)
        word_counts = self.df['consumer_complaint_narrative'].str.split().str.len()
        plt.hist(word_counts, bins=30, color='lightcoral', alpha=0.7)
        plt.title('Distribution of Word Counts')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Category-wise text length analysis
        plt.figure(figsize=(10, 6))
        for category in self.df['category'].unique():
            category_data = self.df[self.df['category'] == category]['consumer_complaint_narrative'].str.len()
            plt.hist(category_data, alpha=0.6, label=self.category_mapping[category], bins=20)
        plt.xlabel('Text Length (characters)')
        plt.ylabel('Frequency')
        plt.title('Text Length Distribution by Category')
        plt.legend()
        plt.savefig('category_text_length.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nCategory distribution:")
        for category, count in category_counts.items():
            print(f"{self.category_mapping[category]}: {count} ({count/len(self.df)*100:.1f}%)")
    
    def preprocess_text(self, text):
        """Preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def text_preprocessing(self):
        """Apply text preprocessing to the dataset"""
        print("\n=== TEXT PREPROCESSING ===")
        
        # Apply preprocessing
        print("Applying text preprocessing...")
        self.df['processed_text'] = self.df['consumer_complaint_narrative'].apply(self.preprocess_text)
        
        # Remove empty texts after preprocessing
        self.df = self.df[self.df['processed_text'].str.len() > 0]
        
        print(f"Dataset shape after preprocessing: {self.df.shape}")
        
        # Show sample of processed text
        print("\nSample of processed text:")
        for i in range(3):
            print(f"\nOriginal: {self.df['consumer_complaint_narrative'].iloc[i][:100]}...")
            print(f"Processed: {self.df['processed_text'].iloc[i][:100]}...")
    
    def feature_engineering(self):
        """Perform feature engineering"""
        print("\n=== FEATURE ENGINEERING ===")
        
        # Text length features
        self.df['text_length'] = self.df['consumer_complaint_narrative'].str.len()
        self.df['word_count'] = self.df['consumer_complaint_narrative'].str.split().str.len()
        self.df['sentence_count'] = self.df['consumer_complaint_narrative'].str.count(r'[.!?]+')
        
        # Average word length
        self.df['avg_word_length'] = self.df['text_length'] / self.df['word_count']
        
        # Keyword features
        keywords = {
            'credit': ['credit', 'score', 'report', 'bureau'],
            'debt': ['debt', 'collect', 'payment', 'owe'],
            'loan': ['loan', 'borrow', 'interest', 'approve'],
            'mortgage': ['mortgage', 'home', 'property', 'house']
        }
        
        for category, words in keywords.items():
            self.df[f'{category}_keywords'] = self.df['consumer_complaint_narrative'].str.lower().str.count('|'.join(words))
        
        print("Feature engineering completed!")
        print(f"New features added: {list(self.df.columns[-8:])}")
    
    def prepare_data(self):
        """Prepare data for modeling"""
        print("\n=== DATA PREPARATION ===")
        
        # Split features and target
        X_text = self.df['processed_text']
        X_features = self.df[['text_length', 'word_count', 'sentence_count', 'avg_word_length',
                            'credit_keywords', 'debt_keywords', 'loan_keywords', 'mortgage_keywords']]
        y = self.df['category']
        
        # Split data
        X_text_train, X_text_test, X_features_train, X_features_test, y_train, y_test = train_test_split(
            X_text, X_features, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set size: {len(X_text_train)}")
        print(f"Test set size: {len(X_text_test)}")
        
        return X_text_train, X_text_test, X_features_train, X_features_test, y_train, y_test
    
    def train_models(self, X_text_train, X_text_test, X_features_train, X_features_test, y_train, y_test):
        """Train and compare multiple models"""
        print("\n=== MODEL TRAINING AND COMPARISON ===")
        
        # Vectorize text data
        print("Vectorizing text data...")
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_text_train_vec = self.vectorizer.fit_transform(X_text_train)
        X_text_test_vec = self.vectorizer.transform(X_text_test)
        
        # Combine text and feature data
        from scipy.sparse import hstack
        X_train_combined = hstack([X_text_train_vec, X_features_train.values])
        X_test_combined = hstack([X_text_test_vec, X_features_test.values])
        
        # Define models
        models = {
            'Naive Bayes': MultinomialNB(),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'SVM': SVC(random_state=42, kernel='linear')
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train_combined, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_combined)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred
            }
            
            print(f"{name} Accuracy: {accuracy:.4f}")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_combined, y_train, cv=5)
            print(f"{name} CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        self.model = results[best_model_name]['model']
        
        print(f"\nBest model: {best_model_name} with accuracy: {results[best_model_name]['accuracy']:.4f}")
        
        return results, y_test
    
    def evaluate_models(self, results, y_test):
        """Evaluate and visualize model performance"""
        print("\n=== MODEL EVALUATION ===")
        
        # Plot accuracy comparison
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        plt.bar(model_names, accuracies, color='skyblue')
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        
        # Confusion matrix for best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_predictions = results[best_model_name]['predictions']
        
        plt.subplot(2, 2, 2)
        cm = confusion_matrix(y_test, best_predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[self.category_mapping[i] for i in sorted(self.category_mapping.keys())],
                   yticklabels=[self.category_mapping[i] for i in sorted(self.category_mapping.keys())])
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Classification report
        plt.subplot(2, 2, 3)
        report = classification_report(y_test, best_predictions, 
                                     target_names=[self.category_mapping[i] for i in sorted(self.category_mapping.keys())],
                                     output_dict=True)
        
        # Extract precision, recall, f1-score for visualization
        metrics = ['precision', 'recall', 'f1-score']
        categories = list(self.category_mapping.values())
        
        x = np.arange(len(categories))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [report[category][metric] for category in categories]
            plt.bar(x + i*width, values, width, label=metric)
        
        plt.xlabel('Categories')
        plt.ylabel('Score')
        plt.title(f'Performance Metrics - {best_model_name}')
        plt.xticks(x + width, categories, rotation=45)
        plt.legend()
        
        # Feature importance (for Random Forest)
        if best_model_name == 'Random Forest':
            plt.subplot(2, 2, 4)
            feature_names = self.vectorizer.get_feature_names_out().tolist() + \
                          ['text_length', 'word_count', 'sentence_count', 'avg_word_length',
                           'credit_keywords', 'debt_keywords', 'loan_keywords', 'mortgage_keywords']
            
            importances = self.model.feature_importances_
            top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:15]
            
            features, scores = zip(*top_features)
            plt.barh(range(len(features)), scores)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Importance')
            plt.title('Top 15 Feature Importances')
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print detailed classification report
        print(f"\nDetailed Classification Report for {best_model_name}:")
        print(classification_report(y_test, best_predictions, 
                                  target_names=[self.category_mapping[i] for i in sorted(self.category_mapping.keys())]))
    
    def predict_new_complaint(self, complaint_text):
        """Predict category for new complaint"""
        if self.model is None or self.vectorizer is None:
            print("Model not trained yet!")
            return None
        
        # Preprocess text
        processed_text = self.preprocess_text(complaint_text)
        
        # Extract features
        text_length = len(complaint_text)
        word_count = len(complaint_text.split())
        sentence_count = complaint_text.count('.') + complaint_text.count('!') + complaint_text.count('?')
        avg_word_length = text_length / word_count if word_count > 0 else 0
        
        # Keyword counts
        keywords = {
            'credit': ['credit', 'score', 'report', 'bureau'],
            'debt': ['debt', 'collect', 'payment', 'owe'],
            'loan': ['loan', 'borrow', 'interest', 'approve'],
            'mortgage': ['mortgage', 'home', 'property', 'house']
        }
        
        keyword_counts = []
        for words in keywords.values():
            count = sum(complaint_text.lower().count(word) for word in words)
            keyword_counts.append(count)
        
        # Vectorize text
        text_vec = self.vectorizer.transform([processed_text])
        
        # Combine features
        features = np.array([[text_length, word_count, sentence_count, avg_word_length] + keyword_counts])
        
        from scipy.sparse import hstack
        combined_features = hstack([text_vec, features])
        
        # Make prediction
        prediction = self.model.predict(combined_features)[0]
        probability = self.model.predict_proba(combined_features)[0] if hasattr(self.model, 'predict_proba') else None
        
        result = {
            'category': self.category_mapping[prediction],
            'category_id': prediction
        }
        
        if probability is not None:
            result['probabilities'] = {
                self.category_mapping[i]: prob for i, prob in enumerate(probability)
            }
        
        return result
    
    def run_complete_analysis(self, file_path):
        """Run the complete analysis pipeline"""
        print("=== CONSUMER COMPLAINT TEXT CLASSIFICATION ===")
        print("Task 5: Kaiburr Assessment")
        print(f"Using real dataset: {file_path}")
        
        # Load data
        self.load_data(file_path)
        
        # Exploratory Data Analysis
        self.exploratory_data_analysis()
        
        # Text Preprocessing
        self.text_preprocessing()
        
        # Feature Engineering
        self.feature_engineering()
        
        # Prepare data
        X_text_train, X_text_test, X_features_train, X_features_test, y_train, y_test = self.prepare_data()
        
        # Train models
        results, y_test = self.train_models(X_text_train, X_text_test, X_features_train, X_features_test, y_train, y_test)
        
        # Evaluate models
        self.evaluate_models(results, y_test)
        
        # Demo prediction
        print("\n=== DEMO PREDICTION ===")
        sample_complaint = "I am having issues with my mortgage payment being processed incorrectly and I need help resolving this matter."
        prediction = self.predict_new_complaint(sample_complaint)
        print(f"Sample complaint: {sample_complaint}")
        print(f"Predicted category: {prediction['category']}")
        if 'probabilities' in prediction:
            print("Category probabilities:")
            for category, prob in prediction['probabilities'].items():
                print(f"  {category}: {prob:.4f}")
        
        print("\n=== ANALYSIS COMPLETE ===")
        return results

if __name__ == "__main__":
    # Initialize classifier
    classifier = ConsumerComplaintClassifier()
    
    # Use the real dataset from D:\csv\complaints.csv
    dataset_path = r"D:\csv\complaints.csv"
    
    # Run complete analysis with real data
    print("Starting Task 5: Consumer Complaint Classification")
    print(f"Using real dataset: {dataset_path}")
    print("=" * 60)
    
    results = classifier.run_complete_analysis(dataset_path)
