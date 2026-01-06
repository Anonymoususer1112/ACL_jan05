"""
Evaluation script for analyzing dialect bias in language models by comparing
embedding spaces and perplexity scores for SAE and AAE text pairs.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm

# Import your ModelInterface
from model_interface import ModelInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class DialectEvaluator:
    """Evaluates dialect bias in language models by comparing SAE and AAE texts."""
    
    def __init__(
        self, 
        model: ModelInterface,
        output_dir: str = "dialect_eval_results",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize evaluator with model and output settings.
        
        Args:
            model: Model implementing the ModelInterface
            output_dir: Directory to save evaluation results
            device: Device to run evaluation on ('cuda' or 'cpu')
        """
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Model loaded on {self.device}")
        logger.info(f"Results will be saved to {self.output_dir}")
        
    def get_embeddings_batch(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Get embeddings for a list of texts using batched processing.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process in each batch
            
        Returns:
            Array of embeddings with shape (len(texts), embedding_dim)
        """
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Getting embeddings"):
            batch_texts = texts[i:i+batch_size]
            
            # Get embeddings for the batch using your model interface
            batch_embeddings = self.model.get_embeddings(batch_texts)
            
            # Convert to numpy if it's a tensor
            if isinstance(batch_embeddings, torch.Tensor):
                batch_embeddings = batch_embeddings.cpu().numpy()
                
            embeddings.append(batch_embeddings)
        
        # Concatenate all batches
        return np.vstack(embeddings)
    
    def calculate_perplexity_batch(self, texts: List[str], batch_size: int = 8) -> List[float]:
        """
        Calculate perplexity for a list of texts in batches.
        
        Args:
            texts: List of text strings
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of perplexity scores for each text
        """
        perplexities = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Calculating perplexity"):
            batch_texts = texts[i:i+batch_size]
            
            # Get log probabilities from your model interface
            batch_log_probs = self.model.get_log_probs(batch_texts)
            
            # Calculate perplexity from log probabilities
            # Assuming log_probs returns average log probability per token
            batch_perplexities = [np.exp(-log_prob) for log_prob in batch_log_probs]
            perplexities.extend(batch_perplexities)
        
        return perplexities
    
    def calculate_js_distance(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> float:
        """
        Calculate Jensen-Shannon distance between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            
        Returns:
            JS distance value
        """
        # Calculate mean embeddings
        mean_emb1 = np.mean(embeddings1, axis=0)
        mean_emb2 = np.mean(embeddings2, axis=0)
        
        # Normalize
        mean_emb1 = mean_emb1 / np.linalg.norm(mean_emb1)
        mean_emb2 = mean_emb2 / np.linalg.norm(mean_emb2)
        
        # Jensen-Shannon distance
        return jensenshannon(mean_emb1, mean_emb2)
    
    def visualize_embeddings(
        self, 
        embeddings1: np.ndarray, 
        embeddings2: np.ndarray, 
        labels: Tuple[str, str] = ("Standard American English", "African American English"),
        n_samples: int = 500,
        filename: str = "dialect_embeddings_tsne.png"
    ) -> str:
        """
        Visualize embeddings using t-SNE and save the plot.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            labels: Labels for the two embedding sets
            n_samples: Max number of samples to visualize
            filename: Output filename for the plot
            
        Returns:
            Path to the saved visualization
        """
        # Sample if there are too many points
        if len(embeddings1) > n_samples:
            idx = np.random.choice(len(embeddings1), n_samples, replace=False)
            emb1_sample = embeddings1[idx]
            emb2_sample = embeddings2[idx]
        else:
            emb1_sample = embeddings1
            emb2_sample = embeddings2
        
        # Combine embeddings for t-SNE
        combined_embeddings = np.vstack([emb1_sample, emb2_sample])
        
        # Apply t-SNE
        logger.info("Applying t-SNE for visualization...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined_embeddings)-1))
        transformed = tsne.fit_transform(combined_embeddings)
        
        # Split back into two groups
        transformed1 = transformed[:len(emb1_sample)]
        transformed2 = transformed[len(emb1_sample):]
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.scatter(transformed1[:, 0], transformed1[:, 1], c='blue', label=labels[0], alpha=0.5)
        plt.scatter(transformed2[:, 0], transformed2[:, 1], c='red', label=labels[1], alpha=0.5)
        plt.legend()
        plt.title('t-SNE visualization of embeddings')
        plt.xlabel('t-SNE dimension 1')
        plt.ylabel('t-SNE dimension 2')
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path)
        plt.close()
        
        return str(output_path)
    
    def plot_perplexity_comparison(
        self, 
        perplexity1: List[float], 
        perplexity2: List[float],
        labels: Tuple[str, str] = ("Standard American English", "African American English"),
        filename: str = "perplexity_comparison.png"
    ) -> str:
        """
        Plot perplexity distributions for both dialects.
        
        Args:
            perplexity1: Perplexity scores for first dialect
            perplexity2: Perplexity scores for second dialect
            labels: Labels for the two dialects
            filename: Output filename for the plot
            
        Returns:
            Path to the saved plot
        """
        plt.figure(figsize=(10, 6))
        
        # Filter out extreme outliers for better visualization
        def filter_outliers(data, m=5):
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            threshold = m * mad
            return [x for x in data if abs(x - median) <= threshold]
        
        perp1_filtered = filter_outliers(perplexity1)
        perp2_filtered = filter_outliers(perplexity2)
        
        # Plot histograms
        plt.hist(perp1_filtered, bins=30, alpha=0.5, label=labels[0])
        plt.hist(perp2_filtered, bins=30, alpha=0.5, label=labels[1])
        
        # Add lines for means
        plt.axvline(np.mean(perplexity1), color='blue', linestyle='dashed', linewidth=2, 
                    label=f'{labels[0]} Mean: {np.mean(perplexity1):.2f}')
        plt.axvline(np.mean(perplexity2), color='red', linestyle='dashed', linewidth=2,
                    label=f'{labels[1]} Mean: {np.mean(perplexity2):.2f}')
        
        plt.legend()
        plt.title('Distribution of Perplexity Scores by Dialect')
        plt.xlabel('Perplexity')
        plt.ylabel('Frequency')
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path)
        plt.close()
        
        return str(output_path)
    
    def evaluate(
        self, 
        data_file: str, 
        sae_column: str, 
        aae_column: str, 
        sample_size: Optional[int] = None,
        batch_size: int = 16
    ) -> Dict[str, Any]:
        """
        Perform comprehensive dialect evaluation.
        
        Args:
            data_file: Path to CSV data file with paired dialect texts
            sae_column: Column name for SAE texts
            aae_column: Column name for AAE texts
            sample_size: Optional number of samples to evaluate (None for all)
            batch_size: Batch size for processing
            
        Returns:
            Dictionary with evaluation results
        """
        # Load the dataset
        df = pd.read_csv(data_file)
        
        # Sample if needed
        if sample_size and sample_size < len(df):
            df = df.sample(sample_size, random_state=42)
        
        # Extract text pairs
        sae_texts = df[sae_column].tolist()
        aae_texts = df[aae_column].tolist()
        
        logger.info(f"Analyzing {len(sae_texts)} text pairs...")
        
        # Get embeddings
        logger.info("1. Calculating embeddings...")
        sae_embeddings = self.get_embeddings_batch(sae_texts, batch_size)
        aae_embeddings = self.get_embeddings_batch(aae_texts, batch_size)
        
        # Calculate perplexity
        logger.info("2. Calculating perplexity...")
        sae_perplexity = self.calculate_perplexity_batch(sae_texts, batch_size)
        aae_perplexity = self.calculate_perplexity_batch(aae_texts, batch_size)
        
        # Analysis
        results = {
            "embedding_similarity": [],
            "sae_perplexity": sae_perplexity,
            "aae_perplexity": aae_perplexity,
            "perplexity_difference": [],
        }
        
        # Calculate embedding similarities between paired sentences
        logger.info("3. Calculating similarities...")
        for i in range(len(sae_embeddings)):
            # Cosine similarity between corresponding SAE and AAE embeddings
            sim = cosine_similarity([sae_embeddings[i]], [aae_embeddings[i]])[0][0]
            results["embedding_similarity"].append(sim)
            
            # Perplexity difference (absolute)
            perp_diff = abs(sae_perplexity[i] - aae_perplexity[i])
            results["perplexity_difference"].append(perp_diff)
        
        # Calculate aggregate results
        aggregate_results = {
            "mean_embedding_similarity": float(np.mean(results["embedding_similarity"])),
            "std_embedding_similarity": float(np.std(results["embedding_similarity"])),
            "mean_sae_perplexity": float(np.mean(results["sae_perplexity"])),
            "mean_aae_perplexity": float(np.mean(results["aae_perplexity"])),
            "perplexity_difference": float(np.mean(results["perplexity_difference"])),
            "perplexity_ratio": float(np.mean(results["aae_perplexity"]) / np.mean(results["sae_perplexity"])),
            "jensen_shannon_distance": float(self.calculate_js_distance(sae_embeddings, aae_embeddings))
        }
        
        # Visualize the embedding space
        viz_path = self.visualize_embeddings(
            sae_embeddings, 
            aae_embeddings, 
            labels=("Standard American English", "African American English")
        )
        
        # Plot perplexity comparison
        perplexity_plot = self.plot_perplexity_comparison(
            results["sae_perplexity"], 
            results["aae_perplexity"]
        )
        
        # Save detailed results
        detailed_results = {
            "metadata": {
                "data_file": data_file,
                "sample_size": len(sae_texts),
                "model": str(self.model),
                "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "aggregate_results": aggregate_results,
            "visualizations": {
                "embedding_tsne": viz_path,
                "perplexity_plot": perplexity_plot
            }
        }
        
        # Save results to file
        results_path = self.output_dir / "evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(detailed_results, f, indent=2)
            
        logger.info(f"Results saved to {results_path}")
        
        # Print aggregate results
        logger.info("\n----- Aggregate Results -----")
        for key, value in aggregate_results.items():
            logger.info(f"{key}: {value:.4f}")
            
        return detailed_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate dialect bias using embedding space and perplexity analysis")
    
    parser.add_argument("--data_file", type=str, required=True, 
                        help="Path to CSV file with paired dialect texts")
    parser.add_argument("--sae_column", type=str, required=True,
                        help="Column name for Standard American English texts")
    parser.add_argument("--aae_column", type=str, required=True,
                        help="Column name for African American English texts")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model or model name")
    parser.add_argument("--output_dir", type=str, default="dialect_eval_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of samples to evaluate (None for all)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for processing")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run evaluation on ('cuda' or 'cpu')")
    
    args = parser.parse_args()
    
    # Import your model implementation here
    from model_loader import load_model
    
    # Load model with your ModelInterface
    model = load_model(args.model_path, device=args.device)
    
    # Create evaluator
    evaluator = DialectEvaluator(
        model=model,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Run evaluation
    evaluator.evaluate(
        data_file=args.data_file,
        sae_column=args.sae_column,
        aae_column=args.aae_column,
        sample_size=args.sample_size,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()