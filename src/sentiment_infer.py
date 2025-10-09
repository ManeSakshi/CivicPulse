#!/usr/bin/env python3#!/usr/bin/env python3#!/usr/bin/env python3#!/usr/bin/env python3

"""

CivicPulse Sentiment Analysis Model Training"""

# Simple yet effective sentiment analysis using TF-IDF + Logistic Regression

"""CivicPulse Sentiment Analysis Model Training""""""



import pandas as pdSimple yet effective sentiment analysis using TF-IDF + Logistic Regression

import numpy as np"""

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import accuracy_score, classification_report, confusion_matriximport pandas as pdTrains DistilBERT on external data, fine-tunes on civic dataTrains DistilBERT on external data, fine-tunes on civic data

from sklearn.pipeline import Pipeline

import pickleimport numpy as np

import os

from datetime import datetimefrom sklearn.feature_extraction.text import TfidfVectorizer""""""



class CivicSentimentModel:from sklearn.linear_model import LogisticRegression

    def __init__(self):

        self.pipeline = Nonefrom sklearn.model_selection import train_test_split, cross_val_score

        self.civic_df = None

        self.external_df = Nonefrom sklearn.metrics import accuracy_score, classification_report, confusion_matrix

        self.label_encoder = {'positive': 2, 'neutral': 1, 'negative': 0}

        self.label_decoder = {2: 'positive', 1: 'neutral', 0: 'negative'}from sklearn.pipeline import Pipelineimport pandas as pdimport pandas as pd

        

    def load_data(self):import pickle

        """Load civic and external data"""

        print("Loading datasets...")import osimport numpy as npimport n        training_args = TrainingArguments(

        

        # Load civic datafrom datetime import datetime

        self.civic_df = pd.read_csv("data/processed/civic_labeled.csv")

        print(f"   Civic data: {len(self.civic_df)} records")import torch            output_dir='./models/phase2_civic',

        

        # Try to load external data for additional trainingclass CivicSentimentModel:

        try:

            external_train = pd.read_csv("data/processed/external/train_external.csv")    def __init__(self):from transformers import (            num_train_epochs=3,

            # Sample external data to reasonable size

            if len(external_train) > 100000:        self.pipeline = None

                external_train = external_train.sample(100000, random_state=42)

            self.external_df = external_train        self.civic_df = None    DistilBertTokenizer, DistilBertForSequenceClassification,            per_device_train_batch_size=8,

            print(f"   External data: {len(self.external_df)} records")

        except FileNotFoundError:        self.external_df = None

            print("   External data not found, using civic data only")

            self.external_df = None        self.label_encoder = {'positive': 2, 'neutral': 1, 'negative': 0}    Trainer, TrainingArguments, DataCollatorWithPadding            per_device_eval_batch_size=8,

            

    def prepare_training_data(self):        self.label_decoder = {2: 'positive', 1: 'neutral', 0: 'negative'}

        """Prepare combined training dataset"""

        print("Preparing training data...")        )            warmup_steps=100,

        

        # Start with civic data    def load_data(self):

        combined_data = []

        combined_labels = []        """Load civic and external data"""from datasets import Dataset            weight_decay=0.01,

        

        # Add civic data        print("📊 Loading datasets...")

        for _, row in self.civic_df.iterrows():

            combined_data.append(row['text'])        from sklearn.metrics import accuracy_score, precision_recall_fscore_support            logging_dir='./logs',

            combined_labels.append(self.label_encoder[row['label']])

                    # Load civic data

        # Add external data if available

        if self.external_df is not None:        self.civic_df = pd.read_csv("data/processed/civic_labeled.csv")from sklearn.model_selection import train_test_split            logging_steps=50,

            for _, row in self.external_df.iterrows():

                if row['label'] in self.label_encoder:        print(f"   Civic data: {len(self.civic_df)} records")

                    combined_data.append(row['text'])

                    combined_labels.append(self.label_encoder[row['label']])        import pickle            eval_strategy="epoch",  # Fixed from evaluation_strategy

                    

        print(f"   Total training samples: {len(combined_data)}")        # Try to load external data for additional training

        

        # Split data        try:import os            save_strategy="epoch",

        X_train, X_test, y_train, y_test = train_test_split(

            combined_data, combined_labels,             external_train = pd.read_csv("data/processed/external/train_external.csv")

            test_size=0.2, 

            random_state=42,            # Sample external data to reasonable sizefrom datetime import datetime            load_best_model_at_end=True,

            stratify=combined_labels

        )            if len(external_train) > 100000:

        

        return X_train, X_test, y_train, y_test                external_train = external_train.sample(100000, random_state=42)            metric_for_best_model="eval_accuracy",

    

    def train_model(self):            self.external_df = external_train

        """Train the sentiment analysis model"""

        print("\nTRAINING SENTIMENT ANALYSIS MODEL")            print(f"   External data: {len(self.external_df)} records")class CivicSentimentModel:        )port torch

        print("=" * 50)

                except FileNotFoundError:

        # Prepare data

        X_train, X_test, y_train, y_test = self.prepare_training_data()            print("   External data not found, using civic data only")    def __init__(self):from sklearn.model_selection import train_test_split

        

        # Create pipeline with TF-IDF and Logistic Regression            self.external_df = None

        self.pipeline = Pipeline([

            ('tfidf', TfidfVectorizer(                    self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

                max_features=10000,

                ngram_range=(1, 2),    def prepare_training_data(self):

                min_df=2,

                max_df=0.95,        """Prepare combined training dataset"""        self.model = Nonefrom transformers import (

                stop_words='english',

                lowercase=True,        print("🔧 Preparing training data...")

                strip_accents='unicode'

            )),                self.civic_df = None    AutoTokenizer, AutoModelForSequenceClassification,

            ('classifier', LogisticRegression(

                random_state=42,        # Start with civic data

                max_iter=1000,

                class_weight='balanced'        combined_data = []        self.external_train_df = None    TrainingArguments, Trainer, pipeline

            ))

        ])        combined_labels = []

        

        print("Training model...")                self.external_test_df = None)

        

        # Train the model        # Add civic data

        self.pipeline.fit(X_train, y_train)

                for _, row in self.civic_df.iterrows():        import os

        # Make predictions

        y_pred = self.pipeline.predict(X_test)            combined_data.append(row['text'])

        

        # Calculate metrics            combined_labels.append(self.label_encoder[row['label']])    def load_data(self):import pickle

        accuracy = accuracy_score(y_test, y_pred)

                    

        print(f"\nTraining Results:")

        print(f"   Accuracy: {accuracy:.4f}")        # Add external data if available        """Load all required datasets"""from datetime import datetime

        print(f"   Training samples: {len(X_train)}")

        print(f"   Test samples: {len(X_test)}")        if self.external_df is not None:

        

        # Detailed classification report            for _, row in self.external_df.iterrows():        print("📊 Loading datasets...")

        print("\nDetailed Performance:")

        target_names = ['negative', 'neutral', 'positive']                if row['label'] in self.label_encoder:

        print(classification_report(y_test, y_pred, target_names=target_names))

                            combined_data.append(row['text'])        class CivicSentimentModel:

        return accuracy, X_test, y_test, y_pred

                        combined_labels.append(self.label_encoder[row['label']])

    def evaluate_civic_focus(self):

        """Evaluate specifically on civic data"""                            # Load civic data    def __init__(self, model_name="distilbert-base-uncased"):

        print("\nCIVIC-SPECIFIC EVALUATION")

        print("=" * 50)        print(f"   Total training samples: {len(combined_data)}")

        

        # Prepare civic-only test data                self.civic_df = pd.read_csv("data/processed/civic_labeled.csv")        self.model_name = model_name

        civic_texts = self.civic_df['text'].tolist()

        civic_labels = [self.label_encoder[label] for label in self.civic_df['label']]        # Split data

        

        # Split civic data for evaluation        X_train, X_test, y_train, y_test = train_test_split(        print(f"   Civic data: {len(self.civic_df)} records")        self.tokenizer = None

        X_civic_train, X_civic_test, y_civic_train, y_civic_test = train_test_split(

            civic_texts, civic_labels,             combined_data, combined_labels, 

            test_size=0.3, 

            random_state=42,            test_size=0.2,                 self.model = None

            stratify=civic_labels

        )            random_state=42,

        

        # Predict on civic test data            stratify=combined_labels        # Load external data (sample for faster training)        self.trainer = None

        civic_predictions = self.pipeline.predict(X_civic_test)

        civic_accuracy = accuracy_score(y_civic_test, civic_predictions)        )

        

        print(f"Civic Domain Performance:")                self.external_train_df = pd.read_csv("data/processed/external/train_external.csv").sample(50000, random_state=42)        

        print(f"   Civic Accuracy: {civic_accuracy:.4f}")

        print(f"   Civic Test Samples: {len(X_civic_test)}")        return X_train, X_test, y_train, y_test

        

        # Show some examples            self.external_test_df = pd.read_csv("data/processed/external/test_external.csv").sample(10000, random_state=42)    def load_data(self, use_external=True):

        print(f"\nSample Civic Predictions:")

        for i in range(min(5, len(X_civic_test))):    def train_model(self):

            text = X_civic_test[i][:100] + "..." if len(X_civic_test[i]) > 100 else X_civic_test[i]

            actual = self.label_decoder[y_civic_test[i]]        """Train the sentiment analysis model"""        print(f"   External train: {len(self.external_train_df)} records")        """Load training and testing data"""

            predicted = self.label_decoder[civic_predictions[i]]

            status = "[CORRECT]" if actual == predicted else "[WRONG]"        print("\n🚀 TRAINING SENTIMENT ANALYSIS MODEL")

            print(f"   {status} '{text}' -> {predicted} (actual: {actual})")

                    print("=" * 50)        print(f"   External test: {len(self.external_test_df)} records")        print("📊 Loading datasets...")

        return civic_accuracy

            

    def save_model(self, accuracy, civic_accuracy):

        """Save the trained model"""        # Prepare data                

        print("\nSAVING MODEL")

        print("=" * 50)        X_train, X_test, y_train, y_test = self.prepare_training_data()

        

        # Create models directory            def preprocess_text_for_bert(self, texts, labels, max_length=128):        # Load civic data (primary focus)

        os.makedirs("models", exist_ok=True)

                # Create pipeline with TF-IDF and Logistic Regression

        # Save the trained pipeline

        with open('models/sentiment_model.pkl', 'wb') as f:        self.pipeline = Pipeline([        """Tokenize and prepare text for BERT"""        civic_path = "data/processed/civic_labeled.csv"

            pickle.dump(self.pipeline, f)

                        ('tfidf', TfidfVectorizer(

        # Save model metadata

        model_info = {                max_features=10000,        encodings = self.tokenizer(        self.civic_df = pd.read_csv(civic_path)

            'model_type': 'TF-IDF + Logistic Regression',

            'overall_accuracy': accuracy,                ngram_range=(1, 2),

            'civic_accuracy': civic_accuracy,

            'civic_data_size': len(self.civic_df),                min_df=2,            texts.tolist(),        print(f"   Civic data: {len(self.civic_df)} records")

            'external_data_size': len(self.external_df) if self.external_df is not None else 0,

            'label_encoder': self.label_encoder,                max_df=0.95,

            'label_decoder': self.label_decoder,

            'trained_timestamp': datetime.now().isoformat(),                stop_words='english',            truncation=True,        

            'features': 'TF-IDF (max 10k features, 1-2 grams)'

        }                lowercase=True,

        

        with open('models/model_info.pkl', 'wb') as f:                strip_accents='unicode'            padding=True,        if use_external:

            pickle.dump(model_info, f)

                        )),

        print("Model saved successfully!")

        print("   Main model: models/sentiment_model.pkl")            ('classifier', LogisticRegression(            max_length=max_length,            # Load external data for pre-training

        print("   Model info: models/model_info.pkl")

                        random_state=42,

    def run_complete_training(self):

        """Execute complete training pipeline"""                max_iter=1000,            return_tensors='pt'            train_path = "data/processed/external/train_external.csv"

        print("CIVICPULSE SENTIMENT MODEL TRAINING")

        print("=" * 60)                class_weight='balanced'

        

        # Load data            ))        )            test_path = "data/processed/external/test_external.csv"

        self.load_data()

                ])

        # Train model

        accuracy, X_test, y_test, y_pred = self.train_model()                            

        

        # Evaluate on civic data specifically        print("🏋️  Training model...")

        civic_accuracy = self.evaluate_civic_focus()

                        # Create dataset            if os.path.exists(train_path):

        # Save model

        self.save_model(accuracy, civic_accuracy)        # Train the model

        

        print("\nTRAINING COMPLETE!")        self.pipeline.fit(X_train, y_train)        dataset = Dataset.from_dict({                self.external_train = pd.read_csv(train_path, nrows=50000)  # Limit for speed

        print("=" * 60)

        print(f"Overall Accuracy: {accuracy:.4f}")        

        print(f"Civic Accuracy: {civic_accuracy:.4f}")

        print("Model ready for production use!")        # Make predictions            'input_ids': encodings['input_ids'],                self.external_test = pd.read_csv(test_path, nrows=10000)

        

        return accuracy, civic_accuracy        y_pred = self.pipeline.predict(X_test)



def main():                    'attention_mask': encodings['attention_mask'],                print(f"   External train: {len(self.external_train)} records")

    """Main training function"""

    model = CivicSentimentModel()        # Calculate metrics

    model.run_complete_training()

        accuracy = accuracy_score(y_test, y_pred)            'labels': labels.tolist()                print(f"   External test: {len(self.external_test)} records")

if __name__ == "__main__":

    main()        

        print(f"\n📊 Training Results:")        })            else:

        print(f"   Accuracy: {accuracy:.4f}")

        print(f"   Training samples: {len(X_train)}")                        print("   [WARN] External data not found, using civic only")

        print(f"   Test samples: {len(X_test)}")

                return dataset                use_external = False

        # Detailed classification report

        print("\n📈 Detailed Performance:")            

        target_names = ['negative', 'neutral', 'positive']

        print(classification_report(y_test, y_pred, target_names=target_names))    def compute_metrics(self, eval_pred):        self.use_external = use_external

        

        return accuracy, X_test, y_test, y_pred        """Compute evaluation metrics"""        

    

    def evaluate_civic_focus(self):        predictions, labels = eval_pred    def prepare_labels(self):

        """Evaluate specifically on civic data"""

        print("\n🎯 CIVIC-SPECIFIC EVALUATION")        predictions = np.argmax(predictions, axis=1)        """Convert string labels to numeric"""

        print("=" * 50)

                        self.label_map = {'negative': 0, 'neutral': 1, 'positive': 2}

        # Prepare civic-only test data

        civic_texts = self.civic_df['text'].tolist()        accuracy = accuracy_score(labels, predictions)        self.id_to_label = {0: 'negative', 1: 'neutral', 2: 'positive'}

        civic_labels = [self.label_encoder[label] for label in self.civic_df['label']]

                precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')        

        # Split civic data for evaluation

        X_civic_train, X_civic_test, y_civic_train, y_civic_test = train_test_split(                # Convert civic labels

            civic_texts, civic_labels, 

            test_size=0.3,         return {        self.civic_df['labels'] = self.civic_df['label'].map(self.label_map)

            random_state=42,

            stratify=civic_labels            'accuracy': accuracy,        

        )

                    'f1': f1,        if self.use_external:

        # Predict on civic test data

        civic_predictions = self.pipeline.predict(X_civic_test)            'precision': precision,            # Convert external labels

        civic_accuracy = accuracy_score(y_civic_test, civic_predictions)

                    'recall': recall            self.external_train['labels'] = self.external_train['label'].map(self.label_map)

        print(f"📊 Civic Domain Performance:")

        print(f"   Civic Accuracy: {civic_accuracy:.4f}")        }            self.external_test['labels'] = self.external_test['label'].map(self.label_map)

        print(f"   Civic Test Samples: {len(X_civic_test)}")

                        

        # Show some examples

        print(f"\n📝 Sample Civic Predictions:")    def train_simplified_model(self):    def tokenize_data(self, texts, labels, max_length=128):

        for i in range(min(5, len(X_civic_test))):

            text = X_civic_test[i][:100] + "..." if len(X_civic_test[i]) > 100 else X_civic_test[i]        """Simplified training approach - direct fine-tuning on civic data"""        """Tokenize text data for BERT"""

            actual = self.label_decoder[y_civic_test[i]]

            predicted = self.label_decoder[civic_predictions[i]]        print("\n🚀 SIMPLIFIED CIVIC SENTIMENT TRAINING")        encodings = self.tokenizer(

            status = "✅" if actual == predicted else "❌"

            print(f"   {status} '{text}' -> {predicted} (actual: {actual})")        print("=" * 50)            list(texts), truncation=True, padding=True,

            

        return civic_accuracy                    max_length=max_length, return_tensors='pt'

    

    def save_model(self, accuracy, civic_accuracy):        # Initialize model        )

        """Save the trained model"""

        print("\n💾 SAVING MODEL")        self.model = DistilBertForSequenceClassification.from_pretrained(        

        print("=" * 50)

                    'distilbert-base-uncased',        class Dataset:

        # Create models directory

        os.makedirs("models", exist_ok=True)            num_labels=3  # positive, neutral, negative            def __init__(self, encodings, labels):

        

        # Save the trained pipeline        )                self.encodings = encodings

        with open('models/sentiment_model.pkl', 'wb') as f:

            pickle.dump(self.pipeline, f)                        self.labels = labels

            

        # Save model metadata        # Prepare label mapping                

        model_info = {

            'model_type': 'TF-IDF + Logistic Regression',        label_map = {'positive': 2, 'neutral': 1, 'negative': 0}            def __getitem__(self, idx):

            'overall_accuracy': accuracy,

            'civic_accuracy': civic_accuracy,        civic_labels = self.civic_df['label'].map(label_map)                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

            'civic_data_size': len(self.civic_df),

            'external_data_size': len(self.external_df) if self.external_df is not None else 0,                        item['labels'] = torch.tensor(self.labels[idx])

            'label_encoder': self.label_encoder,

            'label_decoder': self.label_decoder,        # Split civic data                return item

            'trained_timestamp': datetime.now().isoformat(),

            'features': 'TF-IDF (max 10k features, 1-2 grams)'        X_train, X_val, y_train, y_val = train_test_split(                

        }

                    self.civic_df['text'],             def __len__(self):

        with open('models/model_info.pkl', 'wb') as f:

            pickle.dump(model_info, f)            civic_labels,                 return len(self.labels)

            

        # Create inference helper            test_size=0.2,                 

        inference_code = f'''

import pickle            random_state=42,        return Dataset(encodings, labels)

import pandas as pd

            stratify=civic_labels        

class CivicSentimentInference:

    def __init__(self):        )    def train_phase1_external(self):

        with open('models/sentiment_model.pkl', 'rb') as f:

            self.pipeline = pickle.load(f)                """Phase 1: Pre-train on external data"""

        with open('models/model_info.pkl', 'rb') as f:

            self.info = pickle.load(f)        print(f"   Training on {len(X_train)} civic records")        if not self.use_external:

        self.label_decoder = self.info['label_decoder']

            print(f"   Validating on {len(X_val)} civic records")            print("⏭️  Skipping external pre-training")

    def predict(self, text):

        """Predict sentiment for a single text"""                    return

        if isinstance(text, str):

            texts = [text]        # Prepare datasets            

        else:

            texts = text        train_dataset = self.preprocess_text_for_bert(X_train, y_train)        print("\n🏗️  PHASE 1: Pre-training on External Data")

            

        predictions = self.pipeline.predict(texts)        eval_dataset = self.preprocess_text_for_bert(X_val, y_val)        print("=" * 50)

        probabilities = self.pipeline.predict_proba(texts)

                        

        results = []

        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):        # Training arguments        # Initialize tokenizer and model

            results.append({{

                'text': texts[i] if i < len(texts) else '',        training_args = TrainingArguments(        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

                'sentiment': self.label_decoder[pred],

                'confidence': max(probs),            output_dir='./models/civic_sentiment',        self.model = AutoModelForSequenceClassification.from_pretrained(

                'probabilities': {{

                    'negative': probs[0],            num_train_epochs=3,            self.model_name, num_labels=3

                    'neutral': probs[1], 

                    'positive': probs[2]            per_device_train_batch_size=8,        )

                }}

            }})            per_device_eval_batch_size=8,        

        

        return results[0] if isinstance(text, str) else results            warmup_steps=100,        # Prepare external datasets

    

    def get_model_info(self):            weight_decay=0.01,        train_dataset = self.tokenize_data(

        """Get model information"""

        return self.info            logging_dir='./logs',            self.external_train['text'].values,



# Example usage:            logging_steps=50,            self.external_train['labels'].values

# inference = CivicSentimentInference()

# result = inference.predict("The road conditions in Sangli are terrible")            eval_strategy="epoch",        )

# print(result['sentiment'], result['confidence'])

'''            save_strategy="epoch",        

        

        with open('models/inference_helper.py', 'w') as f:            load_best_model_at_end=True,        test_dataset = self.tokenize_data(

            f.write(inference_code)

                        metric_for_best_model="eval_accuracy",            self.external_test['text'].values, 

        print("✅ Model saved successfully!")

        print("   Main model: models/sentiment_model.pkl")        )            self.external_test['labels'].values

        print("   Model info: models/model_info.pkl")

        print("   Inference helper: models/inference_helper.py")                )

        

    def run_complete_training(self):        # Initialize trainer        

        """Execute complete training pipeline"""

        print("🚀 CIVICPULSE SENTIMENT MODEL TRAINING")        trainer = Trainer(        # Training arguments

        print("=" * 60)

                    model=self.model,        training_args = TrainingArguments(

        # Load data

        self.load_data()            args=training_args,            output_dir='./models/phase1_external',

        

        # Train model            train_dataset=train_dataset,            num_train_epochs=2,

        accuracy, X_test, y_test, y_pred = self.train_model()

                    eval_dataset=eval_dataset,            per_device_train_batch_size=16,

        # Evaluate on civic data specifically

        civic_accuracy = self.evaluate_civic_focus()            compute_metrics=self.compute_metrics,            per_device_eval_batch_size=16,

        

        # Save model            data_collator=DataCollatorWithPadding(self.tokenizer)            warmup_steps=500,

        self.save_model(accuracy, civic_accuracy)

                )            weight_decay=0.01,

        print("\n🎉 TRAINING COMPLETE!")

        print("=" * 60)                    logging_dir='./logs',

        print(f"✅ Overall Accuracy: {accuracy:.4f}")

        print(f"✅ Civic Accuracy: {civic_accuracy:.4f}")        # Train model            logging_steps=100,

        print("✅ Model ready for production use!")

                print("🚀 Starting training...")            eval_strategy="steps",  # Fixed from evaluation_strategy

        return accuracy, civic_accuracy

        trainer.train()            eval_steps=500,

def main():

    """Main training function"""                    save_strategy="steps",

    model = CivicSentimentModel()

    model.run_complete_training()        # Save model            save_steps=1000,



if __name__ == "__main__":        os.makedirs("models", exist_ok=True)            load_best_model_at_end=True,

    main()
        trainer.save_model("models/civic_sentiment")        )

        print("✅ Training completed and saved!")        

                # Initialize trainer

        return trainer        self.trainer = Trainer(

                    model=self.model,

    def evaluate_model(self, trainer):            args=training_args,

        """Evaluate the trained model"""            train_dataset=train_dataset,

        print("\n📊 MODEL EVALUATION")            eval_dataset=test_dataset,

        print("=" * 50)        )

                

        # Evaluate on validation set        # Train model

        eval_results = trainer.evaluate()        print("🚀 Starting external pre-training...")

                self.trainer.train()

        print("🎯 Model Performance:")        

        print(f"   Accuracy: {eval_results['eval_accuracy']:.4f}")        # Save phase 1 model

        print(f"   F1 Score: {eval_results['eval_f1']:.4f}")        os.makedirs("models", exist_ok=True)

        print(f"   Precision: {eval_results['eval_precision']:.4f}")        self.trainer.save_model("models/phase1_external")

        print(f"   Recall: {eval_results['eval_recall']:.4f}")        self.tokenizer.save_pretrained("models/phase1_external")

                print("💾 Phase 1 model saved to models/phase1_external/")

        # Save evaluation results        

        results = {    def train_phase2_civic(self):

            'model_type': 'DistilBERT',        """Phase 2: Fine-tune on civic data"""

            'civic_data_size': len(self.civic_df),        print("\n🎯 PHASE 2: Fine-tuning on Civic Data")

            'final_accuracy': eval_results['eval_accuracy'],        print("=" * 50)

            'final_f1': eval_results['eval_f1'],        

            'final_precision': eval_results['eval_precision'],        # If no external training, initialize fresh model

            'final_recall': eval_results['eval_recall'],        if not self.use_external:

            'timestamp': datetime.now().isoformat()            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        }            self.model = AutoModelForSequenceClassification.from_pretrained(

                        self.model_name, num_labels=3

        with open('models/training_results.pkl', 'wb') as f:            )

            pickle.dump(results, f)        

                    # Split civic data for training/validation

        return results        train_texts, val_texts, train_labels, val_labels = train_test_split(

                self.civic_df['text'].values,

    def save_production_model(self):            self.civic_df['labels'].values,

        """Save the final model for production use"""            test_size=0.2,

        print("\n💾 Saving Production Model")            random_state=42,

        print("=" * 50)            stratify=self.civic_df['labels'].values

                )

        # Save model and tokenizer for production        

        self.model.save_pretrained("models/sentiment_model")        # Tokenize civic data

        self.tokenizer.save_pretrained("models/sentiment_model")        train_dataset = self.tokenize_data(train_texts, train_labels)

                val_dataset = self.tokenize_data(val_texts, val_labels)

        # Create simple pickle version for compatibility        

        model_data = {        # Fine-tuning arguments (more aggressive for small dataset)

            'model_path': "models/sentiment_model",        training_args = TrainingArguments(

            'tokenizer_path': "models/sentiment_model",            output_dir='./models/civic_sentiment',

            'label_map': {0: 'negative', 1: 'neutral', 2: 'positive'},            num_train_epochs=5,

            'trained_on': 'civic_data',            per_device_train_batch_size=8,

            'timestamp': datetime.now().isoformat()            per_device_eval_batch_size=8,

        }            warmup_steps=100,

                    weight_decay=0.01,

        with open('models/sentiment_model.pkl', 'wb') as f:            logging_dir='./logs',

            pickle.dump(model_data, f)            logging_steps=50,

                    evaluation_strategy="epoch",

        print("✅ Production model saved!")            save_strategy="epoch",

        print("   Model files: models/sentiment_model/")            load_best_model_at_end=True,

        print("   Compatibility: models/sentiment_model.pkl")            metric_for_best_model="eval_loss",

                )

    def run_complete_training(self):        

        """Run the complete training pipeline"""        # Update trainer for civic fine-tuning

        print("🚀 CIVICPULSE SENTIMENT MODEL TRAINING")        self.trainer = Trainer(

        print("=" * 60)            model=self.model,

                    args=training_args,

        # Load data            train_dataset=train_dataset,

        self.load_data()            eval_dataset=val_dataset,

                )

        # Train simplified model        

        trainer = self.train_simplified_model()        # Fine-tune on civic data

                print("🏛️ Starting civic fine-tuning...")

        # Evaluate model        self.trainer.train()

        results = self.evaluate_model(trainer)        

                # Save final model

        # Save for production        self.trainer.save_model("models/civic_sentiment")

        self.save_production_model()        self.tokenizer.save_pretrained("models/civic_sentiment")

                print("💾 Final civic model saved to models/civic_sentiment/")

        print("\n🎉 TRAINING COMPLETE!")        

        print("=" * 60)    def evaluate_model(self):

        print(f"✅ Final Accuracy: {results['final_accuracy']:.4f}")        """Evaluate final model performance"""

        print(f"✅ Final F1 Score: {results['final_f1']:.4f}")        print("\n📊 MODEL EVALUATION")

        print("✅ Model ready for deployment!")        print("=" * 50)

        

def main():        # Create test set from civic data

    """Main training function"""        _, test_texts, _, test_labels = train_test_split(

    model = CivicSentimentModel()            self.civic_df['text'].values,

    model.run_complete_training()            self.civic_df['labels'].values,

            test_size=0.2,

if __name__ == "__main__":            random_state=42,

    main()            stratify=self.civic_df['labels'].values
        )
        
        # Create inference pipeline
        classifier = pipeline(
            "sentiment-analysis",
            model="models/civic_sentiment",
            tokenizer="models/civic_sentiment",
            return_all_scores=True
        )
        
        # Get predictions
        predictions = []
        for text in test_texts:
            result = classifier(text)
            # Get the label with highest score
            pred_label = max(result, key=lambda x: x['score'])['label']
            # Convert LABEL_X back to our format
            if 'LABEL_0' in pred_label or pred_label == 'NEGATIVE':
                predictions.append(0)
            elif 'LABEL_1' in pred_label or pred_label == 'NEUTRAL':
                predictions.append(1)
            else:
                predictions.append(2)
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predictions)
        report = classification_report(
            test_labels, predictions,
            target_names=['negative', 'neutral', 'positive'],
            output_dict=True
        )
        
        print(f"🎯 Accuracy: {accuracy:.3f}")
        print(f"📈 Precision: {report['macro avg']['precision']:.3f}")
        print(f"📈 Recall: {report['macro avg']['recall']:.3f}")
        print(f"📈 F1-Score: {report['macro avg']['f1-score']:.3f}")
        
        # Save evaluation results
        eval_results = {
            'accuracy': accuracy,
            'classification_report': report,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('models/evaluation_results.pkl', 'wb') as f:
            pickle.dump(eval_results, f)
            
        print("💾 Evaluation results saved to models/evaluation_results.pkl")
        
    def train_complete_pipeline(self):
        """Run complete training pipeline"""
        print("🚀 CIVICPULSE SENTIMENT MODEL TRAINING")
        print("=" * 60)
        
        # Step 1: Load and prepare data
        self.load_data(use_external=True)
        self.prepare_labels()
        
        # Step 2: Phase 1 - External pre-training
        self.train_phase1_external()
        
        # Step 3: Phase 2 - Civic fine-tuning  
        self.train_phase2_civic()
        
        # Step 4: Evaluate final model
        self.evaluate_model()
        
        print("\n🎉 TRAINING COMPLETE!")
        print("=" * 60)
        print("✅ Model ready for deployment")
        print("📁 Saved to: models/civic_sentiment/")
        print("🎯 Use this model for real-time civic sentiment analysis!")

def main():
    """Main training function"""
    # Initialize and train model
    model = CivicSentimentModel()
    model.train_complete_pipeline()

if __name__ == "__main__":
    main()
