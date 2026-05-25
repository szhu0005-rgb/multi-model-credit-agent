"""
Train NLP topic models stratified by categorical features (Issue, Company, Sub-issue, etc.)

This script enables deep analysis by training separate NLP models within each category,
allowing discovery of category-specific patterns rather than global patterns.

Usage:
    python train_stratified_models.py --csv data/complaints.csv --groupby Issue --min-samples 500
    python train_stratified_models.py --csv data/complaints.csv --groupby Company --min-samples 500
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
import pickle

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import joblib
from tqdm import tqdm

# Import preprocessing utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.preprocessing import preprocess_batch


class StratifiedTopicModelTrainer:
    """Train separate NLP models for each category within a grouping column."""
    
    def __init__(self, csv_path, groupby_column, min_samples=500, topic_counts=None, 
                 text_column='Consumer complaint narrative'):
        """
        Initialize the trainer.
        
        Args:
            csv_path: Path to CSV file
            groupby_column: Column to group by (Issue, Company, Sub-issue, State, etc.)
            min_samples: Minimum complaints per group to train a model
            topic_counts: List of topic counts to evaluate, e.g. [3, 5, 7]
            text_column: Name of text column to analyze
        """
        self.csv_path = csv_path
        self.groupby_column = groupby_column
        self.min_samples = min_samples
        self.topic_counts = topic_counts or [5, 7, 10, 12]
        self.text_column = text_column
        
        self.df = None
        self.models_metadata = []
        self.group_stats = {}
    
    def load_and_preprocess_data(self):
        """Load CSV and preprocess all complaints."""
        print(f"Loading data from {self.csv_path}...")
        self.df = pd.read_csv(self.csv_path)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Grouping by: {self.groupby_column}")
        
        if self.groupby_column not in self.df.columns:
            raise ValueError(f"Column '{self.groupby_column}' not found in dataset")
        
        # Preprocess all text
        print(f"Preprocessing complaints...")
        self.df['processed_text'] = self.df[self.text_column].apply(
            lambda x: ' '.join(preprocess_batch([str(x)])[0]) if pd.notna(x) else ''
        )
        
        # Filter out empty texts
        self.df = self.df[self.df['processed_text'].str.len() > 0]
        print(f"After preprocessing: {len(self.df)} valid complaints")
    
    def analyze_groups(self):
        """Analyze distribution of data across groups."""
        print(f"\n{'='*80}")
        print(f"GROUP DISTRIBUTION - {self.groupby_column}")
        print(f"{'='*80}\n")
        
        group_counts = self.df[self.groupby_column].value_counts()
        
        trainable = group_counts[group_counts >= self.min_samples]
        skipped = group_counts[group_counts < self.min_samples]
        
        print(f"Total groups: {len(group_counts)}")
        print(f"Groups with ≥{self.min_samples} samples: {len(trainable)} (will train models)")
        print(f"Groups with <{self.min_samples} samples: {len(skipped)} (will skip)")
        
        if len(trainable) > 0:
            print(f"\nTop groups to model:")
            for i, (group, count) in enumerate(trainable.head(10).items(), 1):
                pct = (count / len(self.df)) * 100
                print(f"  {i:2d}. {str(group):<50} {count:>8,} ({pct:>6.2f}%)")
        
        self.group_stats = {
            'total_groups': len(group_counts),
            'trainable_groups': len(trainable),
            'skipped_groups': len(skipped),
            'distribution': group_counts.to_dict()
        }
        
        return trainable
    
    def train_lda_models(self, texts):
        """Train LDA models with different topic counts."""
        results = {}
        
        vectorizer = CountVectorizer(
            max_df=0.95, min_df=1, max_features=2000, stop_words='english'
        )
        X = vectorizer.fit_transform(texts)
        
        if X.shape[1] == 0:
            return None, None, {}
        
        for n_topics in self.topic_counts:
            try:
                model = LatentDirichletAllocation(
                    n_components=n_topics, random_state=42, max_iter=20, n_jobs=-1
                )
                model.fit(X)
                perplexity = model.perplexity(X)
                results[n_topics] = {'model': model, 'perplexity': perplexity}
            except Exception as e:
                print(f"    LDA with {n_topics} topics failed: {e}")
        
        return results, vectorizer, X
    
    def train_nmf_models(self, texts):
        """Train NMF models with different topic counts."""
        results = {}
        
        vectorizer = TfidfVectorizer(
            max_df=0.95, min_df=1, max_features=2000, stop_words='english'
        )
        X = vectorizer.fit_transform(texts)
        
        if X.shape[1] == 0:
            return None, None, {}
        
        for n_topics in self.topic_counts:
            try:
                model = NMF(
                    n_components=n_topics, init='nndsvda', random_state=42, max_iter=500
                )
                model.fit(X)
                reconstruction_error = model.reconstruction_err_
                results[n_topics] = {'model': model, 'score': reconstruction_error}
            except Exception as e:
                print(f"    NMF with {n_topics} topics failed: {e}")
        
        return results, vectorizer, X
    
    def select_best_model(self, lda_results, nmf_results):
        """Select best model by reconstruction/perplexity score."""
        best_model = None
        best_vectorizer = None
        best_n_topics = None
        best_score = float('inf')
        best_type = None
        
        # Check NMF results (lower reconstruction error is better)
        if nmf_results:
            for n_topics, result in nmf_results.items():
                score = result['score']
                if score < best_score:
                    best_score = score
                    best_model = result['model']
                    best_n_topics = n_topics
                    best_type = 'NMF'
        
        # Check LDA results (lower perplexity is better)
        if lda_results:
            for n_topics, result in lda_results.items():
                score = result['perplexity']
                if score < best_score:
                    best_score = score
                    best_model = result['model']
                    best_n_topics = n_topics
                    best_type = 'LDA'
        
        return best_model, best_n_topics, best_score, best_type
    
    def train_group_models(self):
        """Train models for each group."""
        trainable_groups = self.analyze_groups()
        
        if len(trainable_groups) == 0:
            print(f"\nNo groups with ≥{self.min_samples} samples. Increase data or lower threshold.")
            return
        
        output_base = Path(self.csv_path).parent / 'stratified_models'
        output_base.mkdir(exist_ok=True)
        
        groupby_dir = output_base / self.groupby_column.replace(' ', '_')
        groupby_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"TRAINING MODELS FOR EACH GROUP")
        print(f"{'='*80}\n")
        
        for group_name, group_count in tqdm(trainable_groups.items(), desc='Groups'):
            try:
                group_data = self.df[self.df[self.groupby_column] == group_name]
                texts = group_data['processed_text'].tolist()
                
                # Train models
                lda_results, lda_vec, lda_X = self.train_lda_models(texts)
                nmf_results, nmf_vec, nmf_X = self.train_nmf_models(texts)
                
                if not lda_results and not nmf_results:
                    continue
                
                # Select best
                best_model, best_n_topics, best_score, best_type = self.select_best_model(
                    lda_results, nmf_results
                )
                
                # Get vectorizer for best model
                best_vectorizer = nmf_vec if best_type == 'NMF' else lda_vec
                
                # Save model and vectorizer
                safe_name = str(group_name).replace('/', '_').replace(' ', '_')[:50]
                model_path = groupby_dir / f'{safe_name}_model.pkl'
                vec_path = groupby_dir / f'{safe_name}_vectorizer.pkl'
                
                joblib.dump(best_model, model_path)
                joblib.dump(best_vectorizer, vec_path)
                
                # Save metadata
                metadata = {
                    'group_name': str(group_name),
                    'groupby_column': self.groupby_column,
                    'model_type': best_type,
                    'n_topics': best_n_topics,
                    'score': float(best_score),
                    'n_samples': group_count,
                    'vectorizer_type': 'TfidfVectorizer' if best_type == 'NMF' else 'CountVectorizer',
                    'model_path': str(model_path),
                    'vectorizer_path': str(vec_path),
                    'trained_at': datetime.now().isoformat(),
                }
                
                self.models_metadata.append(metadata)
                
            except Exception as e:
                print(f"  Error training model for '{group_name}': {e}")
        
        # Save metadata index
        metadata_path = groupby_dir / 'models_index.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.models_metadata, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"Models saved to: {groupby_dir}")
        print(f"Trained {len(self.models_metadata)} models")
        print(f"Metadata saved to: {metadata_path}")
        
        return groupby_dir, self.models_metadata
    
    def generate_report(self, output_dir):
        """Generate summary report."""
        report_path = Path(output_dir) / 'training_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("STRATIFIED NLP MODEL TRAINING REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Training Configuration:\n")
            f.write(f"  Groupby Column: {self.groupby_column}\n")
            f.write(f"  Minimum Samples per Group: {self.min_samples}\n")
            f.write(f"  Topic Counts Evaluated: {self.topic_counts}\n")
            f.write(f"  Total Complaints: {len(self.df)}\n\n")
            
            f.write(f"Group Statistics:\n")
            f.write(f"  Total Groups: {self.group_stats['total_groups']}\n")
            f.write(f"  Trainable Groups: {self.group_stats['trainable_groups']}\n")
            f.write(f"  Skipped Groups: {self.group_stats['skipped_groups']}\n\n")
            
            f.write(f"Trained Models:\n")
            for i, meta in enumerate(self.models_metadata, 1):
                f.write(f"\n  {i}. {meta['group_name']}\n")
                f.write(f"     Model Type: {meta['model_type']}\n")
                f.write(f"     Topics: {meta['n_topics']}\n")
                f.write(f"     Score: {meta['score']:.4f}\n")
                f.write(f"     Samples: {meta['n_samples']}\n")
        
        return report_path


def main():
    parser = argparse.ArgumentParser(
        description='Train stratified NLP models grouped by categorical features'
    )
    parser.add_argument('--csv', default='data/complaints-2026-05-17_11_33.csv',
                       help='Path to CSV file')
    parser.add_argument('--groupby', default='Issue',
                       help='Column to group by (Issue, Company, Sub-issue, State, etc.)')
    parser.add_argument('--min-samples', type=int, default=500,
                       help='Minimum complaints per group to train a model')
    parser.add_argument('--topics', default='5,7,10,12',
                       help='Topic counts to evaluate (comma-separated)')
    
    args = parser.parse_args()
    topic_counts = [int(t) for t in args.topics.split(',')]
    
    trainer = StratifiedTopicModelTrainer(
        csv_path=args.csv,
        groupby_column=args.groupby,
        min_samples=args.min_samples,
        topic_counts=topic_counts
    )
    
    trainer.load_and_preprocess_data()
    output_dir, metadata = trainer.train_group_models()
    
    if metadata:
        report_path = trainer.generate_report(output_dir)
        print(f"Report saved to: {report_path}")


if __name__ == '__main__':
    main()
