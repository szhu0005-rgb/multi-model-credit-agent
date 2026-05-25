"""
Batch inference script.

This script applies the trained topic model to all complaints in the dataset
and saves results to a CSV file with topic assignments.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.inference import load_model

warnings.filterwarnings('ignore')


def run_batch_inference(csv_path, output_dir=None, batch_size=256):
    """
    Apply topic model to all complaints in a CSV file.
    
    Args:
        csv_path: Path to CSV file with complaints
        output_dir: Directory to save results (default: data/)
        batch_size: Number of texts to process at once
    """
    
    csv_path = Path(csv_path)
    
    if output_dir is None:
        output_dir = csv_path.parent
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("BATCH INFERENCE: APPLYING TOPIC MODEL TO ALL COMPLAINTS")
    print("=" * 80)
    
    # Load data
    print(f"\n📂 Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} rows")
    
    # Get text column - explicitly use complaint narrative
    text_col = 'Consumer complaint narrative'
    if text_col not in df.columns:
        # Fallback to longest text column if not found
        text_cols = df.select_dtypes(include='object').columns
        text_col = max(text_cols, key=lambda col: df[col].astype(str).str.len().mean())
    print(f"✓ Text column: {text_col}")
    
    # Load model
    print(f"\n🤖 Loading trained model...")
    model = load_model()
    print(f"✓ Model loaded: {model.metadata['model_type']} with {model.n_topics} topics")
    
    # Run inference
    print(f"\n⏳ Running inference on {len(df)} texts...")
    print(f"   (using batch size: {batch_size})")
    
    all_topics = []
    all_confidences = []
    topic_distributions = []
    
    texts = df[text_col].tolist()
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Batch Progress"):
        batch_texts = texts[i:i+batch_size]
        
        # Get predictions
        results = model.get_primary_topics_batch(batch_texts)
        
        for result in results:
            all_topics.append(result['topic'])
            all_confidences.append(result['confidence'])
            topic_distributions.append(result['distribution'])
    
    # Create output dataframe
    print(f"\n💾 Creating output dataframe...")
    
    output_df = df.copy()
    output_df['primary_topic'] = all_topics
    output_df['topic_confidence'] = all_confidences
    
    # Add topic probabilities for each topic
    for topic_id in range(model.n_topics):
        topic_probs = [dist[topic_id] for dist in topic_distributions]
        output_df[f'topic_{topic_id}_prob'] = topic_probs
    
    # Save results
    output_file = output_dir / 'complaints_with_topics.csv'
    print(f"✓ Saving results to {output_file}...")
    output_df.to_csv(output_file, index=False)
    
    # Print statistics
    print("\n" + "=" * 80)
    print("INFERENCE STATISTICS")
    print("=" * 80)
    
    print(f"\n📊 Topic Distribution:\n")
    topic_counts = pd.Series(all_topics).value_counts().sort_index()
    for topic_id, count in topic_counts.items():
        percentage = 100 * count / len(all_topics)
        bar_length = int(percentage / 2)
        bar = '█' * bar_length + '░' * (50 - bar_length)
        print(f"Topic {topic_id:2d}: {bar} {count:6d} ({percentage:5.1f}%)")
    
    print(f"\n📈 Confidence Statistics:\n")
    confidence_stats = pd.Series(all_confidences).describe()
    print(f"  - Mean: {confidence_stats['mean']:.4f}")
    print(f"  - Std:  {confidence_stats['std']:.4f}")
    print(f"  - Min:  {confidence_stats['min']:.4f}")
    print(f"  - 25%:  {confidence_stats['25%']:.4f}")
    print(f"  - 50%:  {confidence_stats['50%']:.4f}")
    print(f"  - 75%:  {confidence_stats['75%']:.4f}")
    print(f"  - Max:  {confidence_stats['max']:.4f}")
    
    # High confidence predictions
    high_conf_count = sum(1 for c in all_confidences if c >= 0.5)
    print(f"\n✓ Predictions with confidence ≥ 0.5: {high_conf_count} ({100*high_conf_count/len(all_confidences):.1f}%)\n")
    
    # Save topic summary
    summary_file = output_dir / 'topic_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("TOPIC MODELING INFERENCE SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Model Type: {model.metadata['model_type']}\n")
        f.write(f"Number of Topics: {model.n_topics}\n")
        f.write(f"Documents Analyzed: {len(df)}\n")
        f.write(f"Output File: {output_file}\n\n")
        
        f.write("TOPIC DISTRIBUTION:\n")
        f.write("-" * 80 + "\n")
        for topic_id, count in topic_counts.items():
            percentage = 100 * count / len(all_topics)
            f.write(f"Topic {topic_id}: {count:6d} ({percentage:5.1f}%)\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("TOP TERMS PER TOPIC:\n")
        f.write("-" * 80 + "\n\n")
        
        all_top_terms = model.get_all_top_terms(n_terms=10)
        for topic_id, terms in all_top_terms.items():
            f.write(f"Topic {topic_id}:\n")
            for term, weight in terms:
                f.write(f"  - {term:<20} ({weight:.4f})\n")
            f.write("\n")
    
    print(f"✓ Topic summary saved to {summary_file}")
    
    print("\n" + "=" * 80)
    print("✓ BATCH INFERENCE COMPLETE")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  - {output_file}")
    print(f"  - {summary_file}")
    
    return output_df


def main(csv_filename=None, output_suffix=None):
    """Main batch inference pipeline.
    
    Args:
        csv_filename: Optional CSV filename to use (e.g., 'complaints-2026-05-17_11_33.csv')
        output_suffix: Optional suffix for output directory (e.g., '2026-05-17')
    """
    
    # Configuration
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / 'data'
    
    if csv_filename:
        csv_path = data_dir / csv_filename
        # Extract date from filename for output directory
        if output_suffix is None and len(csv_filename) > 20:
            output_suffix = csv_filename.split('complaints-')[1].split('.csv')[0]
    else:
        csv_path = list(data_dir.glob('*.csv'))[0]
    
    if output_suffix:
        output_dir = data_dir / f'results_{output_suffix}'
    else:
        output_dir = data_dir / 'results'
    
    run_batch_inference(csv_path, output_dir=output_dir)


if __name__ == '__main__':
    import sys
    csv_filename = sys.argv[1] if len(sys.argv) > 1 else None
    output_suffix = sys.argv[2] if len(sys.argv) > 2 else None
    main(csv_filename, output_suffix)
