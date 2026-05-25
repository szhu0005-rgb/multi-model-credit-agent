"""
Stratified Batch Inference - Apply Issue-Level Models to All Complaints

This script applies the 5 trained Issue-level NLP models to all complaints,
assigning topics based on the complaint's Issue category.

Output:
- CSV file with all original columns + stratified topic predictions
- Summary statistics comparing with original global model results
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm

# Import preprocessing utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.preprocessing import preprocess_batch


class StratifiedBatchInference:
    """Apply stratified models to all complaints."""
    
    def __init__(self, csv_path, models_dir, groupby_column='Issue', text_column='Consumer complaint narrative'):
        """
        Initialize inference engine.
        
        Args:
            csv_path: Path to complaints CSV
            models_dir: Path to stratified_models directory
            groupby_column: Column used for stratification (Issue, Company, etc.)
            text_column: Name of text column to analyze
        """
        self.csv_path = csv_path
        self.models_dir = Path(models_dir)
        self.groupby_column = groupby_column
        self.text_column = text_column
        
        self.df = None
        self.models_data = {}  # group_name -> {model, vectorizer, metadata}
        self.results = None
    
    def load_data(self):
        """Load complaints dataset."""
        print(f"Loading data from {self.csv_path}...")
        self.df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.df):,} complaints")
    
    def load_stratified_models(self):
        """Load all stratified models and metadata."""
        groupby_dir = self.models_dir / self.groupby_column.replace(' ', '_')
        metadata_path = groupby_dir / 'models_index.json'
        
        print(f"\nLoading stratified models from {groupby_dir}...")
        
        with open(metadata_path, 'r') as f:
            models_metadata = json.load(f)
        
        for meta in models_metadata:
            group_name = meta['group_name']
            
            try:
                model = joblib.load(meta['model_path'])
                vectorizer = joblib.load(meta['vectorizer_path'])
                
                self.models_data[group_name] = {
                    'model': model,
                    'vectorizer': vectorizer,
                    'metadata': meta,
                    'n_topics': meta['n_topics']
                }
                
                print(f"  ✓ Loaded {group_name} ({meta['n_topics']} topics)")
            except Exception as e:
                print(f"  ✗ Error loading {group_name}: {e}")
        
        print(f"\nSuccessfully loaded {len(self.models_data)} models")
    
    def run_batch_inference(self, batch_size=256):
        """
        Apply stratified models to all complaints.
        
        For each complaint, identifies its group (Issue) and uses that group's model.
        """
        print(f"\n{'='*80}")
        print(f"STRATIFIED BATCH INFERENCE")
        print(f"{'='*80}\n")
        
        # Initialize results dataframe
        results_list = []
        
        # Process complaints in batches by group for efficiency
        for group_name in tqdm(self.models_data.keys(), desc='Processing groups'):
            # Get complaints in this group
            group_mask = self.df[self.groupby_column] == group_name
            group_indices = np.where(group_mask.values)[0]
            group_df = self.df.loc[group_indices].copy()
            
            if len(group_df) == 0:
                continue
            
            # Load model and vectorizer
            model = self.models_data[group_name]['model']
            vectorizer = self.models_data[group_name]['vectorizer']
            n_topics = self.models_data[group_name]['n_topics']
            
            # Preprocess texts
            texts = group_df[self.text_column].astype(str).tolist()
            processed_texts = preprocess_batch(texts)
            processed_texts = [' '.join(tokens) for tokens in processed_texts]
            
            # Vectorize
            X = vectorizer.transform(processed_texts)
            
            # Get topic predictions
            topic_dist = model.transform(X)
            primary_topics = topic_dist.argmax(axis=1)
            primary_confidence = topic_dist.max(axis=1)
            
            # Add to results
            for idx, row_idx in enumerate(group_indices):
                result = {
                    'complaint_index': row_idx,
                    'stratified_primary_topic': primary_topics[idx],
                    'stratified_confidence': primary_confidence[idx]
                }
                
                # Add all topic probabilities
                for topic_id in range(n_topics):
                    result[f'stratified_topic_{topic_id}_prob'] = topic_dist[idx, topic_id]
                
                results_list.append(result)
        
        # Merge results with original data
        results_df = pd.DataFrame(results_list)
        results_df = results_df.sort_values('complaint_index').reset_index(drop=True)
        
        # Merge with original data
        self.results = self.df.copy()
        self.results = self.results.assign(
            stratified_primary_topic=results_df['stratified_primary_topic'].values,
            stratified_confidence=results_df['stratified_confidence'].values
        )
        
        # Add all topic probability columns
        topic_cols = [col for col in results_df.columns if 'topic_' in col and '_prob' in col]
        for col in topic_cols:
            self.results[col] = results_df[col].values
        
        print(f"\nInference complete. Results shape: {self.results.shape}")
    
    def save_results(self, output_dir=None, suffix='stratified'):
        """Save results to CSV."""
        if output_dir is None:
            output_dir = Path(self.csv_path).parent / 'results'
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save results CSV
        csv_filename = f'complaints_with_topics_{suffix}.csv'
        csv_path = output_dir / csv_filename
        self.results.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")
        
        # Generate summary
        summary_path = output_dir / f'topic_summary_{suffix}.txt'
        self._generate_summary(summary_path)
        
        return csv_path, summary_path
    
    def _generate_summary(self, output_path):
        """Generate human-readable summary."""
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("STRATIFIED TOPIC PREDICTIONS - SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total Complaints: {len(self.results):,}\n")
            f.write(f"Stratification Level: {self.groupby_column}\n")
            f.write(f"Analysis Date: {datetime.now().isoformat()}\n\n")
            
            # Topic distribution by Issue
            f.write("Topic Distribution by Issue Category:\n")
            f.write("-" * 80 + "\n")
            
            for group_name in sorted(self.models_data.keys()):
                group_mask = self.results[self.groupby_column] == group_name
                group_results = self.results[group_mask]
                
                f.write(f"\n{group_name}:\n")
                f.write(f"  Total Complaints: {len(group_results):,}\n")
                f.write(f"  Mean Confidence: {group_results['stratified_confidence'].mean():.4f}\n")
                f.write(f"  Min Confidence: {group_results['stratified_confidence'].min():.4f}\n")
                f.write(f"  Max Confidence: {group_results['stratified_confidence'].max():.4f}\n")
                
                # Topic counts for this group
                n_topics = self.models_data[group_name]['n_topics']
                f.write(f"  Topic Distribution ({n_topics} topics):\n")
                
                topic_col = 'stratified_primary_topic'
                topic_counts = group_results[topic_col].value_counts().sort_index()
                
                for topic_id, count in topic_counts.items():
                    pct = (count / len(group_results)) * 100
                    f.write(f"    Topic {topic_id}: {count:>8,} ({pct:>6.2f}%)\n")
            
            # Overall statistics
            f.write(f"\n{'='*80}\n")
            f.write("OVERALL STATISTICS\n")
            f.write(f"{'='*80}\n\n")
            
            f.write(f"Mean Confidence: {self.results['stratified_confidence'].mean():.4f}\n")
            f.write(f"Median Confidence: {self.results['stratified_confidence'].median():.4f}\n")
            f.write(f"Std Dev Confidence: {self.results['stratified_confidence'].std():.4f}\n\n")
            
            # Confidence distribution
            f.write("Confidence Score Distribution:\n")
            conf_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
            conf_labels = ['0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5+']
            f.write("-" * 40 + "\n")
            
            for i, (low, high) in enumerate(zip(conf_bins[:-1], conf_bins[1:])):
                mask = (self.results['stratified_confidence'] >= low) & (self.results['stratified_confidence'] < high)
                count = mask.sum()
                pct = (count / len(self.results)) * 100
                f.write(f"  {conf_labels[i]:<10} {count:>8,} ({pct:>6.2f}%)\n")
        
        print(f"Summary saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Apply stratified NLP models to all complaints'
    )
    parser.add_argument('--csv', default='data/complaints-2026-05-17_11_33.csv',
                       help='Path to complaints CSV')
    parser.add_argument('--models-dir', default='data/stratified_models',
                       help='Path to stratified_models directory')
    parser.add_argument('--groupby', default='Issue',
                       help='Grouping column for stratification')
    parser.add_argument('--output-dir', default='data/results',
                       help='Output directory for results')
    parser.add_argument('--suffix', default='stratified_issue',
                       help='Suffix for output files')
    
    args = parser.parse_args()
    
    # Run inference
    inference = StratifiedBatchInference(
        csv_path=args.csv,
        models_dir=args.models_dir,
        groupby_column=args.groupby
    )
    
    inference.load_data()
    inference.load_stratified_models()
    inference.run_batch_inference()
    
    csv_path, summary_path = inference.save_results(
        output_dir=args.output_dir,
        suffix=args.suffix
    )
    
    print(f"\n{'='*80}")
    print("STRATIFIED BATCH INFERENCE COMPLETE")
    print(f"{'='*80}")
    print(f"Results CSV: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == '__main__':
    main()
