"""
Supervised Learning Trainer for PDF Parameter Extraction

This module trains models using PDFs with their corresponding Excel answer sheets.
The model learns from the correct answers and can verify its predictions.
"""

import logging
import pandas as pd
import numpy as np
import json
import pickle
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

from .pdf_processor import PDFProcessor
from .training_data_extractor import TrainingDataExtractor

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """Single training example with PDF and correct answers."""
    pdf_path: str
    pdf_filename: str
    extracted_text: str
    correct_answers: Dict[str, Any]
    predicted_answers: Dict[str, Any] = None
    accuracy_scores: Dict[str, float] = None
    

@dataclass
class ModelPerformance:
    """Model performance metrics."""
    overall_accuracy: float
    parameter_accuracies: Dict[str, float]
    confusion_matrices: Dict[str, np.ndarray]
    improvement_suggestions: List[str]


class SupervisedTrainer:
    """
    Supervised trainer that learns from PDFs with Excel answer sheets.
    """
    
    def __init__(self):
        """Initialize the supervised trainer."""
        self.pdf_processor = PDFProcessor()
        self.extractor = TrainingDataExtractor()
        
        # Model components for each parameter
        self.parameter_models = {}
        self.feature_extractors = {}
        
        # Training data storage
        self.training_examples: List[TrainingExample] = []
        self.validation_examples: List[TrainingExample] = []
        
        # Performance tracking
        self.training_history = []
        
    def load_training_data(self, pdf_directory: str, excel_answers_path: str) -> int:
        """
        Load training data from PDFs and Excel answers.
        
        Args:
            pdf_directory: Directory containing PDF files
            excel_answers_path: Path to Excel file with correct answers
            
        Returns:
            Number of training examples loaded
        """
        print("📚 Loading training data...")
        
        # Load Excel answers
        try:
            answers_df = pd.read_excel(excel_answers_path)
            print(f"✅ Loaded Excel answers: {len(answers_df)} rows")
        except Exception as e:
            logger.error(f"Error loading Excel answers: {str(e)}")
            raise
        
        # Validate Excel columns
        required_columns = ['file_name', 'date', 'company_name', 'company_address', 'angebot']
        missing_columns = [col for col in required_columns if col not in answers_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in Excel: {missing_columns}")
        
        # Load PDFs and match with answers
        pdf_dir = Path(pdf_directory)
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        loaded_examples = 0
        
        for pdf_file in pdf_files:
            # Find corresponding answer in Excel
            answer_row = answers_df[answers_df['file_name'] == pdf_file.name]
            
            if answer_row.empty:
                print(f"⚠️  No answer found for {pdf_file.name}, skipping...")
                continue
            
            try:
                # Extract text from PDF
                processing_result = self.pdf_processor.process_pdf(str(pdf_file))
                
                if not processing_result.success:
                    print(f"❌ Failed to process {pdf_file.name}: {processing_result.error_message}")
                    continue
                
                # Get full text
                extracted_text = " ".join([chunk.content for chunk in processing_result.text_chunks])
                
                # Get correct answers from Excel
                answer_data = answer_row.iloc[0]
                correct_answers = {
                    'date': str(answer_data.get('date', '')).strip() if pd.notna(answer_data.get('date')) else None,
                    'company_name': str(answer_data.get('company_name', '')).strip() if pd.notna(answer_data.get('company_name')) else None,
                    'company_address': str(answer_data.get('company_address', '')).strip() if pd.notna(answer_data.get('company_address')) else None,
                    'angebot': str(answer_data.get('angebot', '')).strip() if pd.notna(answer_data.get('angebot')) else None,
                    'tables': answer_data.get('tables_count', 0) if pd.notna(answer_data.get('tables_count')) else 0
                }
                
                # Create training example
                example = TrainingExample(
                    pdf_path=str(pdf_file),
                    pdf_filename=pdf_file.name,
                    extracted_text=extracted_text,
                    correct_answers=correct_answers
                )
                
                self.training_examples.append(example)
                loaded_examples += 1
                
                if loaded_examples % 10 == 0:
                    print(f"📄 Loaded {loaded_examples} training examples...")
                
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {str(e)}")
                continue
        
        print(f"✅ Loaded {loaded_examples} training examples total")
        
        # Split into training and validation
        if loaded_examples > 1:
            train_examples, val_examples = train_test_split(
                self.training_examples, 
                test_size=0.2, 
                random_state=42
            )
            self.training_examples = train_examples
            self.validation_examples = val_examples
            
            print(f"📊 Split: {len(self.training_examples)} training, {len(self.validation_examples)} validation")
        
        return loaded_examples
    
    def create_features(self, text: str, parameter: str) -> np.ndarray:
        """
        Create features for a specific parameter from text.
        
        Args:
            text: Extracted text from PDF
            parameter: Parameter name (date, company_name, etc.)
            
        Returns:
            Feature vector
        """
        features = []
        
        # Text statistics
        features.extend([
            len(text),
            len(text.split()),
            len(text.split('\n')),
            text.count(','),
            text.count('.'),
            text.count(':'),
        ])
        
        # Parameter-specific features
        if parameter == 'date':
            features.extend([
                len(re.findall(r'\d{1,2}[./]\d{1,2}[./]\d{4}', text)),
                len(re.findall(r'\d{4}[./]\d{1,2}[./]\d{1,2}', text)),
                text.lower().count('datum'),
                text.lower().count('date'),
            ])
        
        elif parameter == 'company_name':
            features.extend([
                text.count('GmbH') + text.count('AG') + text.count('KG'),
                text.count('Inc') + text.count('Ltd') + text.count('Corp'),
                len(re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', text)),
                text.count('&') + text.count('und'),
            ])
        
        elif parameter == 'company_address':
            features.extend([
                len(re.findall(r'\d{5}', text)),  # Postal codes
                text.lower().count('straße') + text.lower().count('str.'),
                text.lower().count('street') + text.lower().count('avenue'),
                len(re.findall(r'\d+[a-z]?', text)),  # House numbers
            ])
        
        elif parameter == 'angebot':
            features.extend([
                text.lower().count('angebot'),
                text.lower().count('quote'),
                text.lower().count('quotation'),
                len(re.findall(r'[A-Z]-\d{4}-\d{3}', text)),  # Quote patterns
            ])
        
        # Position-based features (where in document the parameter might appear)
        text_parts = text.split('\n')
        first_quarter = '\n'.join(text_parts[:len(text_parts)//4])
        last_quarter = '\n'.join(text_parts[3*len(text_parts)//4:])
        
        # Check if parameter indicators appear in first or last quarter
        param_indicators = {
            'date': ['datum', 'date'],
            'company_name': ['gmbh', 'ag', 'inc', 'ltd'],
            'company_address': ['straße', 'str.', 'street'],
            'angebot': ['angebot', 'quote']
        }
        
        indicators = param_indicators.get(parameter, [])
        features.extend([
            sum(first_quarter.lower().count(ind) for ind in indicators),
            sum(last_quarter.lower().count(ind) for ind in indicators),
        ])
        
        return np.array(features)
    
    def extract_parameter_value(self, text: str, parameter: str) -> Optional[str]:
        """
        Extract parameter value from text using trained patterns.
        
        Args:
            text: Extracted text from PDF
            parameter: Parameter name
            
        Returns:
            Extracted parameter value
        """
        if parameter == 'date':
            patterns = [
                r'\b\d{1,2}[./]\d{1,2}[./]\d{4}\b',
                r'\b\d{4}[./]\d{1,2}[./]\d{1,2}\b',
                r'\b\d{1,2}\.\s*\w+\s*\d{4}\b',
            ]
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    return matches[0]
        
        elif parameter == 'company_name':
            patterns = [
                r'\b[\w\s&]+\s+(GmbH|AG|KG|OHG|UG|e\.V\.)\b',
                r'\b[\w\s&]+\s+(Inc|LLC|Ltd|Corp|Corporation|Company)\b',
            ]
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    return matches[0]
        
        elif parameter == 'company_address':
            patterns = [
                r'\b\d{5}\s+[A-Za-zäöüÄÖÜß\s]+\b',
                r'\b[A-Za-zäöüÄÖÜß\s]+str\.\s*\d+[a-z]?\b',
            ]
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    return matches[0]
        
        elif parameter == 'angebot':
            patterns = [
                r'(Angebot|Quote|Quotation)\s*[Nr\.#:]*\s*([A-Za-z0-9\-_]+)',
                r'([A-Z]-\d{4}-\d{3})',
            ]
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    if isinstance(matches[0], tuple):
                        return matches[0][1] if len(matches[0]) > 1 else matches[0][0]
                    else:
                        return matches[0]
        
        return None
    
    def train_parameter_model(self, parameter: str) -> Dict[str, Any]:
        """
        Train a model for a specific parameter.
        
        Args:
            parameter: Parameter name to train
            
        Returns:
            Training results
        """
        print(f"🤖 Training model for {parameter}...")
        
        # Prepare training data
        X_train = []
        y_train = []
        
        for example in self.training_examples:
            features = self.create_features(example.extracted_text, parameter)
            X_train.append(features)
            
            # Label: 1 if parameter exists and is correct, 0 otherwise
            correct_value = example.correct_answers.get(parameter)
            predicted_value = self.extract_parameter_value(example.extracted_text, parameter)
            
            # Simple similarity check
            if correct_value and predicted_value:
                if parameter == 'tables':
                    label = 1 if int(correct_value) > 0 else 0
                else:
                    # Check if predicted contains key parts of correct answer
                    correct_clean = str(correct_value).lower().strip()
                    predicted_clean = str(predicted_value).lower().strip()
                    
                    if parameter == 'date':
                        # Extract numbers from both dates
                        correct_nums = re.findall(r'\d+', correct_clean)
                        predicted_nums = re.findall(r'\d+', predicted_clean)
                        label = 1 if len(set(correct_nums) & set(predicted_nums)) >= 2 else 0
                    else:
                        # General string similarity
                        label = 1 if (correct_clean in predicted_clean or 
                                    predicted_clean in correct_clean or
                                    len(set(correct_clean.split()) & set(predicted_clean.split())) > 0) else 0
            else:
                label = 1 if (correct_value and predicted_value) else 0
            
            y_train.append(label)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        print(f"   📊 Training data: {len(X_train)} examples")
        print(f"   📈 Positive examples: {sum(y_train)}/{len(y_train)} ({sum(y_train)/len(y_train)*100:.1f}%)")
        
        # Train classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Store model
        self.parameter_models[parameter] = model
        
        # Evaluate on validation set if available
        val_accuracy = 0.0
        if self.validation_examples:
            X_val = []
            y_val = []
            
            for example in self.validation_examples:
                features = self.create_features(example.extracted_text, parameter)
                X_val.append(features)
                
                correct_value = example.correct_answers.get(parameter)
                predicted_value = self.extract_parameter_value(example.extracted_text, parameter)
                
                # Same labeling logic as training
                if correct_value and predicted_value:
                    if parameter == 'tables':
                        label = 1 if int(correct_value) > 0 else 0
                    else:
                        correct_clean = str(correct_value).lower().strip()
                        predicted_clean = str(predicted_value).lower().strip()
                        
                        if parameter == 'date':
                            correct_nums = re.findall(r'\d+', correct_clean)
                            predicted_nums = re.findall(r'\d+', predicted_clean)
                            label = 1 if len(set(correct_nums) & set(predicted_nums)) >= 2 else 0
                        else:
                            label = 1 if (correct_clean in predicted_clean or 
                                        predicted_clean in correct_clean or
                                        len(set(correct_clean.split()) & set(predicted_clean.split())) > 0) else 0
                else:
                    label = 1 if (correct_value and predicted_value) else 0
                
                y_val.append(label)
            
            X_val = np.array(X_val)
            y_val = np.array(y_val)
            
            # Predict and calculate accuracy
            y_pred = model.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_pred)
        
        results = {
            'parameter': parameter,
            'training_examples': len(X_train),
            'positive_examples': int(sum(y_train)),
            'training_accuracy': model.score(X_train, y_train),
            'validation_accuracy': val_accuracy,
            'feature_importance': model.feature_importances_.tolist()
        }
        
        print(f"   ✅ Training accuracy: {results['training_accuracy']:.3f}")
        if val_accuracy > 0:
            print(f"   ✅ Validation accuracy: {val_accuracy:.3f}")
        
        return results
    
    def train_all_models(self) -> Dict[str, Any]:
        """
        Train models for all parameters.
        
        Returns:
            Complete training results
        """
        print("🚀 Training models for all parameters...")
        
        if not self.training_examples:
            raise ValueError("No training examples loaded. Call load_training_data() first.")
        
        parameters = ['date', 'company_name', 'company_address', 'angebot', 'tables']
        results = {}
        
        for parameter in parameters:
            try:
                param_results = self.train_parameter_model(parameter)
                results[parameter] = param_results
            except Exception as e:
                logger.error(f"Error training {parameter} model: {str(e)}")
                results[parameter] = {'error': str(e)}
        
        # Store training history
        self.training_history.append({
            'timestamp': pd.Timestamp.now(),
            'results': results
        })
        
        print("✅ Training completed for all parameters!")
        return results
    
    def verify_predictions(self, pdf_path: str, expected_answers: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify model predictions against expected answers.
        
        Args:
            pdf_path: Path to PDF file
            expected_answers: Expected answers from Excel
            
        Returns:
            Verification results with accuracy scores
        """
        print(f"🔍 Verifying predictions for {Path(pdf_path).name}...")
        
        # Extract text from PDF
        processing_result = self.pdf_processor.process_pdf(pdf_path)
        if not processing_result.success:
            return {'error': f"Failed to process PDF: {processing_result.error_message}"}
        
        extracted_text = " ".join([chunk.content for chunk in processing_result.text_chunks])
        
        # Get predictions
        predictions = {}
        verification_results = {}
        
        parameters = ['date', 'company_name', 'company_address', 'angebot', 'tables']
        
        for parameter in parameters:
            # Extract parameter value
            predicted_value = self.extract_parameter_value(extracted_text, parameter)
            predictions[parameter] = predicted_value
            
            # Get expected value
            expected_value = expected_answers.get(parameter)
            
            # Calculate accuracy
            if expected_value and predicted_value:
                if parameter == 'tables':
                    accuracy = 1.0 if int(expected_value) > 0 and predicted_value else 0.0
                else:
                    # String similarity check
                    expected_clean = str(expected_value).lower().strip()
                    predicted_clean = str(predicted_value).lower().strip()
                    
                    if parameter == 'date':
                        expected_nums = re.findall(r'\d+', expected_clean)
                        predicted_nums = re.findall(r'\d+', predicted_clean)
                        accuracy = len(set(expected_nums) & set(predicted_nums)) / max(len(expected_nums), len(predicted_nums)) if expected_nums else 0.0
                    else:
                        # Word overlap similarity
                        expected_words = set(expected_clean.split())
                        predicted_words = set(predicted_clean.split())
                        
                        if expected_words and predicted_words:
                            accuracy = len(expected_words & predicted_words) / len(expected_words | predicted_words)
                        else:
                            accuracy = 0.0
            else:
                accuracy = 1.0 if (not expected_value and not predicted_value) else 0.0
            
            verification_results[parameter] = {
                'expected': expected_value,
                'predicted': predicted_value,
                'accuracy': accuracy,
                'match': accuracy > 0.5
            }
        
        # Overall accuracy
        overall_accuracy = np.mean([r['accuracy'] for r in verification_results.values()])
        
        result = {
            'pdf_file': Path(pdf_path).name,
            'overall_accuracy': overall_accuracy,
            'parameter_results': verification_results,
            'predictions': predictions,
            'success': True
        }
        
        print(f"   📊 Overall accuracy: {overall_accuracy:.3f}")
        
        return result
    
    def batch_verify(self, pdf_directory: str, excel_answers_path: str) -> Dict[str, Any]:
        """
        Verify predictions for multiple PDFs against Excel answers.
        
        Args:
            pdf_directory: Directory containing PDF files
            excel_answers_path: Path to Excel with correct answers
            
        Returns:
            Batch verification results
        """
        print("📦 Running batch verification...")
        
        # Load Excel answers
        answers_df = pd.read_excel(excel_answers_path)
        
        # Process all PDFs
        pdf_dir = Path(pdf_directory)
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        verification_results = []
        
        for pdf_file in pdf_files:
            # Find corresponding answer
            answer_row = answers_df[answers_df['file_name'] == pdf_file.name]
            
            if answer_row.empty:
                print(f"⚠️  No answer found for {pdf_file.name}, skipping...")
                continue
            
            # Get expected answers
            answer_data = answer_row.iloc[0]
            expected_answers = {
                'date': str(answer_data.get('date', '')).strip() if pd.notna(answer_data.get('date')) else None,
                'company_name': str(answer_data.get('company_name', '')).strip() if pd.notna(answer_data.get('company_name')) else None,
                'company_address': str(answer_data.get('company_address', '')).strip() if pd.notna(answer_data.get('company_address')) else None,
                'angebot': str(answer_data.get('angebot', '')).strip() if pd.notna(answer_data.get('angebot')) else None,
                'tables': answer_data.get('tables_count', 0) if pd.notna(answer_data.get('tables_count')) else 0
            }
            
            # Verify predictions
            result = self.verify_predictions(str(pdf_file), expected_answers)
            verification_results.append(result)
        
        # Calculate aggregate statistics
        if verification_results:
            overall_accuracies = [r['overall_accuracy'] for r in verification_results if r.get('success')]
            parameter_accuracies = {}
            
            parameters = ['date', 'company_name', 'company_address', 'angebot', 'tables']
            for param in parameters:
                param_accs = [r['parameter_results'][param]['accuracy'] 
                            for r in verification_results 
                            if r.get('success') and param in r['parameter_results']]
                parameter_accuracies[param] = np.mean(param_accs) if param_accs else 0.0
            
            summary = {
                'total_files': len(verification_results),
                'successful_verifications': len(overall_accuracies),
                'average_overall_accuracy': np.mean(overall_accuracies) if overall_accuracies else 0.0,
                'parameter_accuracies': parameter_accuracies,
                'detailed_results': verification_results
            }
        else:
            summary = {
                'total_files': 0,
                'successful_verifications': 0,
                'average_overall_accuracy': 0.0,
                'parameter_accuracies': {},
                'detailed_results': []
            }
        
        print(f"✅ Batch verification completed:")
        print(f"   📊 Files processed: {summary['total_files']}")
        print(f"   📈 Average accuracy: {summary['average_overall_accuracy']:.3f}")
        
        return summary
    
    def save_models(self, output_directory: str):
        """Save trained models to disk."""
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save parameter models
        for parameter, model in self.parameter_models.items():
            model_path = output_dir / f"{parameter}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Save training history
        history_path = output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            # Convert timestamps to strings for JSON serialization
            history_json = []
            for entry in self.training_history:
                entry_copy = entry.copy()
                entry_copy['timestamp'] = entry['timestamp'].isoformat()
                history_json.append(entry_copy)
            json.dump(history_json, f, indent=2)
        
        print(f"💾 Models saved to {output_directory}")
    
    def load_models(self, model_directory: str):
        """Load trained models from disk."""
        model_dir = Path(model_directory)
        
        parameters = ['date', 'company_name', 'company_address', 'angebot', 'tables']
        
        for parameter in parameters:
            model_path = model_dir / f"{parameter}_model.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.parameter_models[parameter] = pickle.load(f)
                print(f"✅ Loaded {parameter} model")
        
        print(f"📂 Models loaded from {model_directory}")
    
    def generate_training_report(self, output_path: str, training_results: Dict[str, Any]):
        """Generate a comprehensive training report."""
        
        with open(output_path, 'w') as f:
            f.write("# Supervised Training Report\n\n")
            f.write("## Training Summary\n\n")
            
            f.write("| Parameter | Training Examples | Positive Examples | Training Accuracy | Validation Accuracy |\n")
            f.write("|-----------|-------------------|-------------------|-------------------|---------------------|\n")
            
            for param, results in training_results.items():
                if 'error' not in results:
                    f.write(f"| {param} | {results['training_examples']} | {results['positive_examples']} | "
                           f"{results['training_accuracy']:.3f} | {results['validation_accuracy']:.3f} |\n")
            
            f.write("\n## Model Performance\n\n")
            
            for param, results in training_results.items():
                if 'error' not in results:
                    f.write(f"### {param.title()} Model\n")
                    f.write(f"- **Training Accuracy**: {results['training_accuracy']:.3f}\n")
                    f.write(f"- **Validation Accuracy**: {results['validation_accuracy']:.3f}\n")
                    f.write(f"- **Training Examples**: {results['training_examples']}\n")
                    f.write(f"- **Positive Examples**: {results['positive_examples']}\n\n")
        
        print(f"📄 Training report saved to {output_path}")


def main():
    """Example usage of the supervised trainer."""
    trainer = SupervisedTrainer()
    
    # Load training data
    pdf_directory = "./training_pdfs"
    excel_answers = "./training_answers.xlsx"
    
    try:
        # Load training data
        num_examples = trainer.load_training_data(pdf_directory, excel_answers)
        
        if num_examples > 0:
            # Train models
            training_results = trainer.train_all_models()
            
            # Save models
            trainer.save_models("./trained_models")
            
            # Generate report
            trainer.generate_training_report("./training_report.md", training_results)
            
            # Verify on same data (you would use different data in practice)
            verification_results = trainer.batch_verify(pdf_directory, excel_answers)
            
            print("🎉 Supervised training completed successfully!")
            
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")


if __name__ == "__main__":
    main()