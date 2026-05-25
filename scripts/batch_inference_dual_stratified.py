"""
Dual-Stratified Batch Inference - Apply Issue + Company Models Together

This script applies both Issue-level and Company-level NLP models to all complaints,
creating a cross-dimensional analysis that shows:
- Which topics appear in each Issue category
- Which topics appear for each Company
- Company-specific handling of each Issue

Output:
- CSV file with original columns + Issue topics + Company topics
- Summary showing Issue × Company topic patterns
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


class DualStratifiedInference:
    """Apply both Issue-level and Company-level models to all complaints."""
    
    def __init__(self, csv_path, models_dir, text_column='Consumer complaint narrative'):
        """
        Initialize dual inference engine.
        
        Args:
            csv_path: Path to complaints CSV
            models_dir: Path to stratified_models directory
            text_column: Name of text column to analyze
        """
        self.csv_path = csv_path
        self.models_dir = Path(models_dir)
        self.text_column = text_column
        
        self.df = None
        self.issue_models = {}  # issue_name -> {model, vectorizer, metadata}
        self.company_models = {}  # company_name -> {model, vectorizer, metadata}
        self.results = None
    
    def load_data(self):
        """Load complaints dataset."""
        print(f"Loading data from {self.csv_path}...")
        self.df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.df):,} complaints")
    
    def load_models(self, stratification_column, models_dict):
        """Load stratified models for a given column."""
        groupby_dir = self.models_dir / stratification_column.replace(' ', '_')
        metadata_path = groupby_dir / 'models_index.json'
        
        print(f"\nLoading {stratification_column} models from {groupby_dir}...")
        
        if not metadata_path.exists():
            print(f"⚠️  Warning: {metadata_path} not found. Models may not be trained yet.")
            return False
        
        try:
            with open(metadata_path, 'r') as f:
                models_metadata = json.load(f)
            
            for meta in models_metadata:
                group_name = meta['group_name']
                
                try:
                    model = joblib.load(meta['model_path'])
                    vectorizer = joblib.load(meta['vectorizer_path'])
                    
                    models_dict[group_name] = {
                        'model': model,
                        'vectorizer': vectorizer,
                        'metadata': meta,
                        'n_topics': meta['n_topics']
                    }
                    
                    print(f"  ✓ {group_name:<50} ({meta['n_topics']} topics)")
                except Exception as e:
                    print(f"  ✗ Error loading {group_name}: {e}")
            
            print(f"Successfully loaded {len(models_dict)} {stratification_column} models")
            return True
        
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def run_dual_inference(self, batch_size=256):
        """
        Apply both Issue and Company models to all complaints.
        """
        print(f"\n{'='*100}")
        print(f"DUAL-STRATIFIED BATCH INFERENCE (Issue × Company)")
        print(f"{'='*100}\n")
        
        if not self.issue_models:
            print("ERROR: Issue models not loaded. Cannot proceed.")
            return False
        
        if not self.company_models:
            print("WARNING: Company models not loaded. Proceeding with Issue models only.")
            company_only = True
        else:
            company_only = False
        
        # Initialize results
        results_list = []
        
        # Process all complaints
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc='Applying models'):
            complaint_text = str(row[self.text_column])
            issue = row['Issue']
            company = row['Company']
            
            result = {'complaint_index': idx}
            
            # === APPLY ISSUE MODEL ===
            if issue in self.issue_models:
                try:
                    model = self.issue_models[issue]['model']
                    vectorizer = self.issue_models[issue]['vectorizer']
                    n_topics = self.issue_models[issue]['n_topics']
                    
                    # Preprocess and vectorize
                    processed = ' '.join(preprocess_batch([complaint_text])[0])
                    X = vectorizer.transform([processed])
                    
                    # Get predictions
                    topic_dist = model.transform(X)[0]
                    primary_topic = topic_dist.argmax()
                    primary_confidence = topic_dist.max()
                    
                    result['issue'] = issue
                    result['issue_primary_topic'] = primary_topic
                    result['issue_confidence'] = primary_confidence
                    
                    # Add all topic probabilities
                    for topic_id in range(n_topics):
                        result[f'issue_topic_{topic_id}_prob'] = topic_dist[topic_id]
                
                except Exception as e:
                    result['issue_error'] = str(e)
            else:
                result['issue_error'] = f"No model for {issue}"
            
            # === APPLY COMPANY MODEL ===
            if not company_only and company in self.company_models:
                try:
                    model = self.company_models[company]['model']
                    vectorizer = self.company_models[company]['vectorizer']
                    n_topics = self.company_models[company]['n_topics']
                    
                    # Preprocess and vectorize
                    processed = ' '.join(preprocess_batch([complaint_text])[0])
                    X = vectorizer.transform([processed])
                    
                    # Get predictions
                    topic_dist = model.transform(X)[0]
                    primary_topic = topic_dist.argmax()
                    primary_confidence = topic_dist.max()
                    
                    result['company'] = company
                    result['company_primary_topic'] = primary_topic
                    result['company_confidence'] = primary_confidence
                    
                    # Add all topic probabilities
                    for topic_id in range(n_topics):
                        result[f'company_topic_{topic_id}_prob'] = topic_dist[topic_id]
                
                except Exception as e:
                    result['company_error'] = str(e)
            elif not company_only:
                result['company_error'] = f"No model for {company}"
            
            results_list.append(result)
        
        # Merge results with original data
        results_df = pd.DataFrame(results_list)
        self.results = self.df.copy()
        
        # Add Issue predictions
        self.results['issue_primary_topic'] = results_df['issue_primary_topic'].values
        self.results['issue_confidence'] = results_df['issue_confidence'].values
        
        issue_topic_cols = [col for col in results_df.columns if col.startswith('issue_topic_') and '_prob' in col]
        for col in issue_topic_cols:
            self.results[col] = results_df[col].values
        
        # Add Company predictions (if available)
        if not company_only:
            self.results['company_primary_topic'] = results_df['company_primary_topic'].values
            self.results['company_confidence'] = results_df['company_confidence'].values
            
            company_topic_cols = [col for col in results_df.columns if col.startswith('company_topic_') and '_prob' in col]
            for col in company_topic_cols:
                self.results[col] = results_df[col].values
        
        print(f"\nInference complete. Results shape: {self.results.shape}")
        return True
    
    def save_results(self, output_dir=None, suffix='dual_stratified'):
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
            f.write("=" * 100 + "\n")
            f.write("DUAL-STRATIFIED TOPIC PREDICTIONS - SUMMARY\n")
            f.write("=" * 100 + "\n\n")
            
            f.write(f"Total Complaints: {len(self.results):,}\n")
            f.write(f"Analysis Date: {datetime.now().isoformat()}\n\n")
            
            # === ISSUE LEVEL SUMMARY ===
            if 'issue_primary_topic' in self.results.columns:
                f.write("ISSUE-LEVEL TOPIC PREDICTIONS\n")
                f.write("-" * 100 + "\n\n")
                
                for issue_name in self.issue_models.keys():
                    issue_mask = self.results['Issue'] == issue_name
                    issue_results = self.results[issue_mask]
                    
                    if len(issue_results) == 0:
                        continue
                    
                    f.write(f"\n{issue_name}:\n")
                    f.write(f"  Total: {len(issue_results):,}\n")
                    f.write(f"  Mean Confidence: {issue_results['issue_confidence'].mean():.4f}\n")
                    f.write(f"  Topic Distribution:\n")
                    
                    topic_counts = issue_results['issue_primary_topic'].value_counts().sort_index()
                    for topic_id, count in topic_counts.items():
                        pct = (count / len(issue_results)) * 100
                        f.write(f"    Topic {topic_id}: {count:>8,} ({pct:>6.2f}%)\n")
            
            # === COMPANY LEVEL SUMMARY ===
            if 'company_primary_topic' in self.results.columns:
                f.write(f"\n{'='*100}\n")
                f.write("COMPANY-LEVEL TOPIC PREDICTIONS\n")
                f.write("-" * 100 + "\n\n")
                
                for company_name in self.company_models.keys():
                    company_mask = self.results['Company'] == company_name
                    company_results = self.results[company_mask]
                    
                    if len(company_results) == 0:
                        continue
                    
                    f.write(f"\n{company_name}:\n")
                    f.write(f"  Total: {len(company_results):,}\n")
                    f.write(f"  Mean Confidence: {company_results['company_confidence'].mean():.4f}\n")
                    f.write(f"  Topic Distribution:\n")
                    
                    topic_counts = company_results['company_primary_topic'].value_counts().sort_index()
                    for topic_id, count in topic_counts.items():
                        pct = (count / len(company_results)) * 100
                        f.write(f"    Topic {topic_id}: {count:>8,} ({pct:>6.2f}%)\n")
            
            # === CROSS-DIMENSIONAL ANALYSIS ===
            if 'issue_primary_topic' in self.results.columns and 'company_primary_topic' in self.results.columns:
                f.write(f"\n{'='*100}\n")
                f.write("ISSUE × COMPANY CROSS-DIMENSIONAL ANALYSIS\n")
                f.write("-" * 100 + "\n\n")
                
                # Show where Issue and Company models agree/disagree
                agreement = (self.results['issue_primary_topic'] == self.results['company_primary_topic']).sum()
                agreement_pct = (agreement / len(self.results)) * 100
                
                f.write(f"Model Agreement: {agreement:,} / {len(self.results):,} ({agreement_pct:.2f}%)\n")
                f.write(f"Model Disagreement: {len(self.results) - agreement:,} / {len(self.results):,} ({100-agreement_pct:.2f}%)\n\n")
                
                f.write("When models disagree, it suggests:\n")
                f.write("- Issue-specific patterns different from company patterns\n")
                f.write("- Company-specific handling of that issue type\n")
                f.write("- Root cause may be both company AND issue related\n")
            
            # === OVERALL STATISTICS ===
            f.write(f"\n{'='*100}\n")
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 100 + "\n\n")
            
            if 'issue_confidence' in self.results.columns:
                f.write(f"Issue Model Confidence:\n")
                f.write(f"  Mean: {self.results['issue_confidence'].mean():.4f}\n")
                f.write(f"  Median: {self.results['issue_confidence'].median():.4f}\n")
                f.write(f"  Std Dev: {self.results['issue_confidence'].std():.4f}\n\n")
            
            if 'company_confidence' in self.results.columns:
                f.write(f"Company Model Confidence:\n")
                f.write(f"  Mean: {self.results['company_confidence'].mean():.4f}\n")
                f.write(f"  Median: {self.results['company_confidence'].median():.4f}\n")
                f.write(f"  Std Dev: {self.results['company_confidence'].std():.4f}\n")
        
        print(f"Summary saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Apply dual-stratified (Issue + Company) NLP models to all complaints'
    )
    parser.add_argument('--csv', default='data/complaints-2026-05-17_11_33.csv',
                       help='Path to complaints CSV')
    parser.add_argument('--models-dir', default='data/stratified_models',
                       help='Path to stratified_models directory')
    parser.add_argument('--output-dir', default='data/results',
                       help='Output directory for results')
    parser.add_argument('--suffix', default='dual_stratified',
                       help='Suffix for output files')
    
    args = parser.parse_args()
    
    # Run inference
    inference = DualStratifiedInference(
        csv_path=args.csv,
        models_dir=args.models_dir
    )
    
    inference.load_data()
    
    # Load both Issue and Company models
    issue_success = inference.load_models('Issue', inference.issue_models)
    company_success = inference.load_models('Company', inference.company_models)
    
    if not issue_success:
        print("ERROR: Issue models required. Cannot proceed.")
        return
    
    if not company_success:
        print("WARNING: Company models not found. Proceeding with Issue models only.")
    
    # Run inference
    if inference.run_dual_inference():
        csv_path, summary_path = inference.save_results(
            output_dir=args.output_dir,
            suffix=args.suffix
        )
        
        print(f"\n{'='*100}")
        print("DUAL-STRATIFIED BATCH INFERENCE COMPLETE")
        print(f"{'='*100}")
        print(f"Results CSV: {csv_path}")
        print(f"Summary: {summary_path}")
    else:
        print("ERROR: Inference failed.")


if __name__ == '__main__':
    main()
