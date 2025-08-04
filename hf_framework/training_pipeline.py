"""
Training Pipeline for preparing and training models on PDF extraction data.

This module handles:
1. Data preparation from labeled Excel files
2. Model training for the 5 specific parameters
3. Data validation and quality assessment
4. Training/validation split
5. Model evaluation and improvement
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import torch
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
import matplotlib.pyplot as plt
import seaborn as sns

from .training_data_extractor import TrainingDataExtractor, TrainingDataPoint
from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    model_name: str = "distilbert-base-multilingual-cased"
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    validation_split: float = 0.2
    test_split: float = 0.1
    save_model_path: str = "./trained_models"
    
    # Data quality thresholds
    min_confidence: float = 0.7
    min_training_samples: int = 10


@dataclass
class TrainingMetrics:
    """Training and validation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score
        }


class DataQualityAssessment:
    """Assess quality of training data."""
    
    def __init__(self):
        self.quality_report = {}
    
    def assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess the quality of training data.
        
        Args:
            df: DataFrame with extracted data
            
        Returns:
            Quality assessment report
        """
        report = {
            'total_samples': len(df),
            'completeness': {},
            'data_types': {},
            'value_distributions': {},
            'quality_score': 0.0,
            'issues': [],
            'recommendations': []
        }
        
        # Check completeness for each field
        fields = ['date', 'company_name', 'company_address', 'tables_count', 'angebot']
        
        for field in fields:
            if field in df.columns:
                non_null_count = df[field].notna().sum()
                completeness = non_null_count / len(df)
                report['completeness'][field] = {
                    'count': int(non_null_count),
                    'percentage': float(completeness)
                }
                
                if completeness < 0.5:
                    report['issues'].append(f"Low completeness for {field}: {completeness:.1%}")
                    report['recommendations'].append(f"Improve extraction patterns for {field}")
        
        # Check for duplicates
        duplicate_count = df.duplicated(subset=['file_name']).sum()
        if duplicate_count > 0:
            report['issues'].append(f"Found {duplicate_count} duplicate files")
            report['recommendations'].append("Remove duplicate entries")
        
        # Assess date formats
        if 'date' in df.columns:
            date_samples = df['date'].dropna().head(20)
            date_formats = self._analyze_date_formats(date_samples.tolist())
            report['value_distributions']['date_formats'] = date_formats
        
        # Assess company name patterns
        if 'company_name' in df.columns:
            company_samples = df['company_name'].dropna()
            report['value_distributions']['company_indicators'] = self._analyze_company_patterns(company_samples.tolist())
        
        # Calculate overall quality score
        avg_completeness = np.mean([v['percentage'] for v in report['completeness'].values()])
        quality_penalties = len(report['issues']) * 0.1
        report['quality_score'] = max(0.0, avg_completeness - quality_penalties)
        
        # Add recommendations based on quality score
        if report['quality_score'] < 0.6:
            report['recommendations'].append("Consider improving PDF extraction patterns")
            report['recommendations'].append("Manual review and correction of extracted data recommended")
        elif report['quality_score'] < 0.8:
            report['recommendations'].append("Data quality is moderate - some manual corrections may help")
        
        self.quality_report = report
        return report
    
    def _analyze_date_formats(self, dates: List[str]) -> Dict[str, int]:
        """Analyze date format patterns."""
        formats = {}
        
        for date_str in dates:
            if not date_str:
                continue
                
            # Classify date format
            if '/' in date_str:
                if date_str.count('/') == 2:
                    formats['DD/MM/YYYY or MM/DD/YYYY'] = formats.get('DD/MM/YYYY or MM/DD/YYYY', 0) + 1
            elif '.' in date_str:
                if date_str.count('.') == 2:
                    formats['DD.MM.YYYY'] = formats.get('DD.MM.YYYY', 0) + 1
            elif '-' in date_str:
                if date_str.count('-') == 2:
                    formats['YYYY-MM-DD'] = formats.get('YYYY-MM-DD', 0) + 1
            else:
                formats['Other'] = formats.get('Other', 0) + 1
        
        return formats
    
    def _analyze_company_patterns(self, companies: List[str]) -> Dict[str, int]:
        """Analyze company name patterns."""
        patterns = {
            'GmbH': 0,
            'AG': 0,
            'Ltd/LLC/Inc': 0,
            'Other': 0
        }
        
        for company in companies[:50]:  # Sample first 50
            if not company:
                continue
                
            company_lower = company.lower()
            
            if 'gmbh' in company_lower:
                patterns['GmbH'] += 1
            elif ' ag' in company_lower or company_lower.endswith('ag'):
                patterns['AG'] += 1
            elif any(indicator in company_lower for indicator in ['ltd', 'llc', 'inc', 'corp']):
                patterns['Ltd/LLC/Inc'] += 1
            else:
                patterns['Other'] += 1
        
        return patterns
    
    def generate_quality_report(self, output_path: str):
        """Generate a detailed quality report."""
        if not self.quality_report:
            return
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Data Quality Assessment Report\n\n")
            
            f.write(f"## Overview\n")
            f.write(f"- Total Samples: {self.quality_report['total_samples']}\n")
            f.write(f"- Quality Score: {self.quality_report['quality_score']:.2f}/1.0\n\n")
            
            f.write(f"## Data Completeness\n")
            for field, stats in self.quality_report['completeness'].items():
                f.write(f"- {field}: {stats['count']}/{self.quality_report['total_samples']} ({stats['percentage']:.1%})\n")
            
            f.write(f"\n## Issues Found\n")
            for issue in self.quality_report['issues']:
                f.write(f"- {issue}\n")
            
            f.write(f"\n## Recommendations\n")
            for rec in self.quality_report['recommendations']:
                f.write(f"- {rec}\n")


class TrainingPipeline:
    """
    Main training pipeline for PDF extraction models.
    """
    
    def __init__(self, config: TrainingConfig = None):
        """Initialize the training pipeline."""
        self.config = config or TrainingConfig()
        self.extractor = TrainingDataExtractor()
        self.quality_assessor = DataQualityAssessment()
        
        # Ensure output directories exist
        Path(self.config.save_model_path).mkdir(parents=True, exist_ok=True)
        
    def prepare_training_data(self, pdf_directory: str, labeled_excel_path: str = None) -> pd.DataFrame:
        """
        Prepare training data from PDFs and labeled Excel file.
        
        Args:
            pdf_directory: Directory containing PDF files
            labeled_excel_path: Optional path to labeled Excel data
            
        Returns:
            Combined training dataset
        """
        logger.info("Preparing training data...")
        
        # Extract data from PDFs
        extracted_data = self.extractor.extract_batch(pdf_directory)
        
        # Convert to DataFrame
        extracted_df = pd.DataFrame([data.to_excel_row() for data in extracted_data])
        
        # Load labeled data if provided
        if labeled_excel_path and Path(labeled_excel_path).exists():
            logger.info(f"Loading labeled data from {labeled_excel_path}")
            labeled_df = self.extractor.load_labeled_data(labeled_excel_path)
            
            # Merge extracted and labeled data
            # Use labeled data as ground truth where available
            combined_df = self._merge_extracted_and_labeled(extracted_df, labeled_df)
        else:
            logger.info("No labeled data provided, using extracted data only")
            combined_df = extracted_df
        
        # Assess data quality
        quality_report = self.quality_assessor.assess_data_quality(combined_df)
        logger.info(f"Data quality score: {quality_report['quality_score']:.2f}")
        
        if quality_report['quality_score'] < 0.5:
            logger.warning("Low data quality detected. Consider manual review.")
        
        return combined_df
    
    def _merge_extracted_and_labeled(self, extracted_df: pd.DataFrame, labeled_df: pd.DataFrame) -> pd.DataFrame:
        """Merge extracted and labeled data, preferring labeled data."""
        
        # Merge on file_name
        merged_df = extracted_df.merge(
            labeled_df, 
            on='file_name', 
            how='left', 
            suffixes=('_extracted', '_labeled')
        )
        
        # Create final columns using labeled data where available
        final_columns = {}
        fields = ['date', 'company_name', 'company_address', 'angebot']
        
        for field in fields:
            extracted_col = f"{field}_extracted"
            labeled_col = f"{field}_labeled"
            
            if labeled_col in merged_df.columns:
                # Use labeled data where available, fall back to extracted
                final_columns[field] = merged_df[labeled_col].fillna(merged_df[extracted_col])
            else:
                final_columns[field] = merged_df[extracted_col]
        
        # Keep other columns
        result_df = merged_df[['file_name']].copy()
        for field, values in final_columns.items():
            result_df[field] = values
        
        # Add metadata columns
        metadata_cols = ['page_count', 'processing_time', 'errors', 'tables_count']
        for col in metadata_cols:
            if f"{col}_extracted" in merged_df.columns:
                result_df[col] = merged_df[f"{col}_extracted"]
        
        return result_df
    
    def create_training_datasets(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create training datasets for each extraction task.
        
        Args:
            df: Combined training dataset
            
        Returns:
            Dictionary containing training datasets for each field
        """
        datasets = {}
        
        # Prepare datasets for each field
        fields = ['date', 'company_name', 'company_address', 'angebot']
        
        for field in fields:
            if field not in df.columns:
                continue
                
            # Filter rows where this field has data
            field_df = df[df[field].notna()].copy()
            
            if len(field_df) < self.config.min_training_samples:
                logger.warning(f"Insufficient training data for {field}: {len(field_df)} samples")
                continue
            
            # Create binary classification: extracted vs not extracted
            # This is a simplified approach - you could make it more sophisticated
            field_df['has_field'] = 1
            
            # Add negative examples (where field is missing)
            negative_df = df[df[field].isna()].copy()
            negative_df['has_field'] = 0
            
            # Combine positive and negative examples
            combined_field_df = pd.concat([field_df, negative_df], ignore_index=True)
            
            # Use raw text as features (you would need to store this during extraction)
            # For now, we'll use file name and other available features
            features = self._create_features(combined_field_df, field)
            labels = combined_field_df['has_field'].values
            
            # Split data
            X_train, X_temp, y_train, y_temp = train_test_split(
                features, labels, 
                test_size=self.config.validation_split + self.config.test_split,
                random_state=42,
                stratify=labels
            )
            
            val_test_split = self.config.test_split / (self.config.validation_split + self.config.test_split)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp,
                test_size=val_test_split,
                random_state=42,
                stratify=y_temp
            )
            
            datasets[field] = {
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test,
                'feature_names': self._get_feature_names(field)
            }
            
            logger.info(f"Created dataset for {field}: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
        
        return datasets
    
    def _create_features(self, df: pd.DataFrame, field: str) -> np.ndarray:
        """Create features for training."""
        
        # Simple feature engineering based on available data
        features = []
        
        for _, row in df.iterrows():
            feature_vector = []
            
            # File name features
            filename = row.get('file_name', '')
            feature_vector.extend([
                len(filename),
                filename.lower().count('angebot'),
                filename.lower().count('quote'),
                filename.lower().count('invoice'),
                filename.lower().count('rechnung')
            ])
            
            # Page count feature
            feature_vector.append(row.get('page_count', 0))
            
            # Tables count feature
            feature_vector.append(row.get('tables_count', 0))
            
            # Processing time (indicator of document complexity)
            feature_vector.append(row.get('processing_time', 0))
            
            # Error indicator
            feature_vector.append(1 if row.get('errors') else 0)
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _get_feature_names(self, field: str) -> List[str]:
        """Get feature names for a field."""
        return [
            'filename_length',
            'filename_angebot_count',
            'filename_quote_count', 
            'filename_invoice_count',
            'filename_rechnung_count',
            'page_count',
            'tables_count',
            'processing_time',
            'has_errors'
        ]
    
    def train_classifiers(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train classifiers for each field.
        
        Args:
            datasets: Training datasets for each field
            
        Returns:
            Trained models and metrics
        """
        trained_models = {}
        
        for field, data in datasets.items():
            logger.info(f"Training classifier for {field}...")
            
            # Train multiple classifiers and choose the best
            classifiers = {
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
            }
            
            best_model = None
            best_score = 0
            best_metrics = None
            
            for clf_name, clf in classifiers.items():
                # Train classifier
                clf.fit(data['X_train'], data['y_train'])
                
                # Evaluate on validation set
                y_pred = clf.predict(data['X_val'])
                
                # Calculate metrics
                accuracy = accuracy_score(data['y_val'], y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    data['y_val'], y_pred, average='binary'
                )
                
                metrics = TrainingMetrics(
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1_score=f1
                )
                
                logger.info(f"  {clf_name}: Accuracy={accuracy:.3f}, F1={f1:.3f}")
                
                # Keep best model
                if f1 > best_score:
                    best_score = f1
                    best_model = clf
                    best_metrics = metrics
            
            # Test best model on test set
            if best_model and len(data['X_test']) > 0:
                y_test_pred = best_model.predict(data['X_test'])
                test_accuracy = accuracy_score(data['y_test'], y_test_pred)
                test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
                    data['y_test'], y_test_pred, average='binary'
                )
                
                test_metrics = TrainingMetrics(
                    accuracy=test_accuracy,
                    precision=test_precision,
                    recall=test_recall,
                    f1_score=test_f1
                )
                
                logger.info(f"Test performance for {field}: Accuracy={test_accuracy:.3f}, F1={test_f1:.3f}")
            else:
                test_metrics = best_metrics
            
            trained_models[field] = {
                'model': best_model,
                'val_metrics': best_metrics,
                'test_metrics': test_metrics,
                'feature_names': data['feature_names']
            }
            
            # Save model
            model_path = Path(self.config.save_model_path) / f"{field}_classifier.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(best_model, f)
            
            logger.info(f"Saved model for {field} to {model_path}")
        
        return trained_models
    
    def generate_training_report(self, trained_models: Dict[str, Any], output_path: str):
        """Generate a comprehensive training report."""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Training Report\n\n")
            
            f.write("## Model Performance Summary\n\n")
            f.write("| Field | Test Accuracy | Test F1 | Test Precision | Test Recall |\n")
            f.write("|-------|---------------|---------|----------------|-------------|\n")
            
            for field, model_data in trained_models.items():
                metrics = model_data['test_metrics']
                f.write(f"| {field} | {metrics.accuracy:.3f} | {metrics.f1_score:.3f} | "
                       f"{metrics.precision:.3f} | {metrics.recall:.3f} |\n")
            
            f.write("\n## Detailed Results\n\n")
            
            for field, model_data in trained_models.items():
                f.write(f"### {field.title()} Extraction\n")
                
                val_metrics = model_data['val_metrics']
                test_metrics = model_data['test_metrics']
                
                f.write(f"**Validation Performance:**\n")
                f.write(f"- Accuracy: {val_metrics.accuracy:.3f}\n")
                f.write(f"- F1 Score: {val_metrics.f1_score:.3f}\n")
                f.write(f"- Precision: {val_metrics.precision:.3f}\n")
                f.write(f"- Recall: {val_metrics.recall:.3f}\n\n")
                
                f.write(f"**Test Performance:**\n")
                f.write(f"- Accuracy: {test_metrics.accuracy:.3f}\n")
                f.write(f"- F1 Score: {test_metrics.f1_score:.3f}\n")
                f.write(f"- Precision: {test_metrics.precision:.3f}\n")
                f.write(f"- Recall: {test_metrics.recall:.3f}\n\n")
        
        logger.info(f"Training report saved to {output_path}")
    
    def run_full_pipeline(self, pdf_directory: str, labeled_excel_path: str = None, 
                         output_dir: str = "./training_output") -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Args:
            pdf_directory: Directory containing PDF files
            labeled_excel_path: Optional path to labeled Excel data
            output_dir: Directory to save outputs
            
        Returns:
            Complete training results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Starting full training pipeline...")
        
        # Step 1: Prepare data
        df = self.prepare_training_data(pdf_directory, labeled_excel_path)
        
        # Save prepared data
        prepared_data_path = output_path / "prepared_training_data.xlsx"
        df.to_excel(prepared_data_path, index=False)
        logger.info(f"Prepared data saved to {prepared_data_path}")
        
        # Step 2: Assess data quality
        quality_report_path = output_path / "data_quality_report.md"
        self.quality_assessor.generate_quality_report(str(quality_report_path))
        
        # Step 3: Create training datasets
        datasets = self.create_training_datasets(df)
        
        if not datasets:
            logger.error("No datasets created - insufficient training data")
            return {}
        
        # Step 4: Train models
        trained_models = self.train_classifiers(datasets)
        
        # Step 5: Generate reports
        training_report_path = output_path / "training_report.md"
        self.generate_training_report(trained_models, str(training_report_path))
        
        # Step 6: Save results
        results = {
            'trained_models': trained_models,
            'datasets': datasets,
            'data_quality': self.quality_assessor.quality_report,
            'config': self.config
        }
        
        results_path = output_path / "training_results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info("Training pipeline completed successfully!")
        logger.info(f"Results saved to {output_dir}")
        
        return results