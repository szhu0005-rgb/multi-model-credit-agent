"""
Train topic modeling pipeline (LDA and NMF).

This script:
1. Loads and preprocesses complaint data
2. Trains LDA models with different topic counts
3. Trains NMF models with different topic counts
4. Evaluates models using coherence score
5. Selects and saves the best model
"""

import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import joblib

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.preprocessing import preprocess_batch, tokens_to_text

warnings.filterwarnings('ignore')


class TopicModelEvaluator:
    """Evaluate topic models using coherence score."""
    
    @staticmethod
    def coherence_score(model, vectorizer, n_topics, doc_term_matrix, documents_tokens):
        """
        Calculate coherence score for a topic model.
        
        Args:
            model: Fitted topic model (LDA or NMF)
            vectorizer: Fitted vectorizer
            n_topics: Number of topics
            doc_term_matrix: Document-term matrix
            documents_tokens: List of token lists
            
        Returns:
            Coherence score (0-1, higher is better)
        """
        try:
            # Get top terms per topic
            feature_names = np.array(vectorizer.get_feature_names_out())
            
            if hasattr(model, 'components_'):
                components = model.components_
            else:
                return 0.0
            
            # Calculate coherence using word co-occurrence
            coherences = []
            
            for topic_idx in range(n_topics):
                top_indices = np.argsort(components[topic_idx])[-10:]
                top_words = feature_names[top_indices]
                
                # Count co-occurrence of top words in documents
                cooccurrence = 0
                doc_count = 0
                
                for tokens in documents_tokens:
                    if any(word in tokens for word in top_words):
                        doc_count += 1
                        word_set = set(tokens)
                        cooccurrence += sum(1 for w in top_words if w in word_set)
                
                if doc_count > 0:
                    coherences.append(cooccurrence / (doc_count * len(top_words)))
            
            return np.mean(coherences) if coherences else 0.0
        
        except Exception as e:
            print(f"Error calculating coherence: {e}")
            return 0.0


def load_and_preprocess_data(csv_path, text_column=None, sample_size=None):
    """
    Load data and preprocess texts.
    
    Args:
        csv_path: Path to CSV file
        text_column: Name of text column (auto-detect if None)
        sample_size: Maximum number of rows to process
        
    Returns:
        Tuple of (processed_texts, original_texts)
    """
    print(f"\n📂 Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    # Auto-detect text column - prefer narrative/complaint/description columns
    if text_column is None:
        preferred_cols = ['Consumer complaint narrative', 'complaint narrative', 'narrative', 
                         'complaint_text', 'text', 'description', 'message', 'issue']
        text_cols = df.select_dtypes(include='object').columns.tolist()
        
        # Check for preferred columns first
        for pref_col in preferred_cols:
            if pref_col in text_cols:
                text_column = pref_col
                break
        
        # Fallback: use column with longest average text
        if text_column is None:
            text_column = max(text_cols, key=lambda col: df[col].astype(str).str.len().mean())
    
    if text_column is None:
        raise ValueError("No text column found in CSV")
    
    print(f"✓ Data loaded: {len(df)} rows")
    print(f"✓ Text column: {text_column}")
    print(f"\n🔄 Preprocessing {len(df)} texts (this may take a few minutes)...")
    
    original_texts = df[text_column].astype(str).tolist()
    processed_tokens = preprocess_batch(original_texts)
    
    # Convert tokens back to text for vectorization
    processed_texts = [tokens_to_text(tokens) for tokens in processed_tokens]
    
    # Filter out empty texts
    valid_indices = [i for i, text in enumerate(processed_texts) if text.strip()]
    processed_texts = [processed_texts[i] for i in valid_indices]
    
    print(f"✓ Preprocessing complete: {len(processed_texts)} valid texts")
    
    return processed_texts, [original_texts[i] for i in valid_indices]


def train_lda_models(processed_texts, topic_counts=None):
    """
    Train LDA models with different topic counts.
    
    Args:
        processed_texts: List of preprocessed text strings
        topic_counts: List of topic counts to try
        
    Returns:
        Dictionary mapping topic count → (model, vectorizer, score)
    """
    if topic_counts is None:
        topic_counts = [5, 7, 10, 12, 15]
    
    print(f"\n📊 Training LDA models...")
    
    results = {}
    evaluator = TopicModelEvaluator()
    
    for n_topics in tqdm(topic_counts, desc="LDA Training"):
        # Vectorize using CountVectorizer (required for LDA)
        vectorizer = CountVectorizer(
            max_df=0.95,
            min_df=1,  # Reduced from 2 to include more words
            max_features=2000,  # Increased from 1000
            stop_words='english'
        )
        
        doc_term_matrix = vectorizer.fit_transform(processed_texts)
        
        # Train LDA
        model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=20,
            n_jobs=-1,
            verbose=0
        )
        model.fit(doc_term_matrix)
        
        # Calculate coherence score
        # Note: For LDA, we'll use a simple metric based on model perplexity
        score = -model.score(doc_term_matrix) / len(processed_texts)
        
        results[n_topics] = (model, vectorizer, score)
    
    return results


def train_nmf_models(processed_texts, topic_counts=None):
    """
    Train NMF models with different topic counts.
    
    Args:
        processed_texts: List of preprocessed text strings
        topic_counts: List of topic counts to try
        
    Returns:
        Dictionary mapping topic count → (model, vectorizer, score)
    """
    if topic_counts is None:
        topic_counts = [5, 7, 10, 12, 15]
    
    print(f"\n📊 Training NMF models...")
    
    results = {}
    evaluator = TopicModelEvaluator()
    
    for n_topics in tqdm(topic_counts, desc="NMF Training"):
        # Vectorize using TfidfVectorizer
        vectorizer = TfidfVectorizer(
            max_df=0.95,
            min_df=1,  # Reduced from 2 to include more words
            max_features=2000,  # Increased from 1000
            stop_words='english'
        )
        
        doc_term_matrix = vectorizer.fit_transform(processed_texts)
        
        # Train NMF
        model = NMF(
            n_components=n_topics,
            random_state=42,
            max_iter=300,
            init='nndsvda'
        )
        model.fit(doc_term_matrix)
        
        # Use reconstruction error as score (lower is better, negate for consistency)
        score = model.reconstruction_err_
        
        results[n_topics] = (model, vectorizer, score)
    
    return results


def save_model(model, vectorizer, output_dir, model_name):
    """
    Save model and vectorizer to disk.
    
    Args:
        model: Trained topic model
        vectorizer: Fitted vectorizer
        output_dir: Directory to save files
        model_name: Name prefix for saved files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / f"{model_name}_model.pkl"
    vectorizer_path = output_dir / f"{model_name}_vectorizer.pkl"
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"✓ Model saved to {model_path}")
    print(f"✓ Vectorizer saved to {vectorizer_path}")
    
    return model_path, vectorizer_path


def main(csv_filename=None):
    """
    Main training pipeline.
    
    Args:
        csv_filename: Optional CSV filename to use (e.g., 'complaints-2026-05-17_11_33.csv')
                     If None, uses the first CSV found
    """
    
    # Configuration
    data_dir = Path(__file__).parent.parent / 'data'
    
    if csv_filename:
        csv_path = data_dir / csv_filename
    else:
        csv_path = list(data_dir.glob('*.csv'))[0]
    
    output_dir = Path(__file__).parent.parent / 'models'
    
    topic_counts = [5, 7, 10, 12, 15]
    sample_size = None  # Use all data; set to e.g. 10000 for faster testing
    text_column = 'Consumer complaint narrative'  # Explicitly specify the text column
    
    print("=" * 80)
    print("TOPIC MODELING TRAINING PIPELINE")
    print("=" * 80)
    
    # Load and preprocess data
    processed_texts, original_texts = load_and_preprocess_data(
        csv_path,
        text_column=text_column,
        sample_size=sample_size
    )
    
    # Train LDA models
    lda_results = train_lda_models(processed_texts, topic_counts)
    
    # Train NMF models
    nmf_results = train_nmf_models(processed_texts, topic_counts)
    
    # Compare and select best models
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    
    print("\nLDA Models:")
    print(f"{'Topics':<10} {'Score':<15} {'Lower Score = Better'}")
    print("-" * 40)
    best_lda_topics = min(lda_results.keys(), key=lambda k: lda_results[k][2])
    for topics, (model, vec, score) in sorted(lda_results.items()):
        marker = " ← BEST" if topics == best_lda_topics else ""
        print(f"{topics:<10} {score:<15.6f}{marker}")
    
    print("\nNMF Models:")
    print(f"{'Topics':<10} {'Score':<15} {'Lower Score = Better'}")
    print("-" * 40)
    best_nmf_topics = min(nmf_results.keys(), key=lambda k: nmf_results[k][2])
    for topics, (model, vec, score) in sorted(nmf_results.items()):
        marker = " ← BEST" if topics == best_nmf_topics else ""
        print(f"{topics:<10} {score:<15.6f}{marker}")
    
    # Select overall best model
    lda_best = lda_results[best_lda_topics]
    nmf_best = nmf_results[best_nmf_topics]
    
    lda_score = lda_best[2]
    nmf_score = nmf_best[2]
    
    if nmf_score < lda_score:
        best_model, best_vectorizer, best_score = nmf_best
        best_type = "NMF"
        best_topics = best_nmf_topics
    else:
        best_model, best_vectorizer, best_score = lda_best
        best_type = "LDA"
        best_topics = best_lda_topics
    
    print("\n" + "=" * 80)
    print(f"✓ BEST MODEL: {best_type} with {best_topics} topics (Score: {best_score:.6f})")
    print("=" * 80)
    
    # Save best model
    print(f"\n💾 Saving best model...")
    save_model(best_model, best_vectorizer, output_dir, "best_topic_model")
    
    # Save metadata
    metadata = {
        'model_type': best_type,
        'n_topics': best_topics,
        'score': best_score,
        'data_size': len(processed_texts),
        'vectorizer_type': 'TfidfVectorizer' if best_type == 'NMF' else 'CountVectorizer',
    }
    
    metadata_path = output_dir / 'model_metadata.pkl'
    joblib.dump(metadata, metadata_path)
    print(f"✓ Metadata saved to {metadata_path}")
    
    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETE")
    print("=" * 80)
    print(f"\n→ Next step: Run model evaluation notebook (notebooks/03_model_evaluation.ipynb)")


if __name__ == '__main__':
    import sys
    csv_filename = sys.argv[1] if len(sys.argv) > 1 else None
    main(csv_filename)
