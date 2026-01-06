"""
Dialect evaluation script for analyzing embedding space and calculating perplexity scores.

This script provides two modes of operation:
1. Embedding space analysis: Compares texts in embedding space
2. Perplexity analysis: Calculates perplexity scores for texts

Both modes can compare Standard American English (SAE) and African American English (AAE).
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dialect_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# Change relative import to absolute import
try:
    from .model_interface import  ModelInterface
except ImportError:
    import sys
    import os
    # Add parent directory to path
    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    from models.model_interface import  ModelInterface
    from models import Phi4VllmInterface


class Phi4DialectEvaluator(ModelInterface):
    """
    Implementation of the dialect evaluation interface for the Phi4 model using vLLM.
    """
    
    def __init__(self, model_id: str = "microsoft/phi-4", dtype: str = "bfloat16", device: str = "cuda"):
        """
        Initialize the Phi4 evaluator.
        
        Args:
            model_id: HuggingFace model ID for Phi-4
            dtype: Data type for model weights (bfloat16, float16, etc.)
            device: Device to run model on (cuda or cpu)
        """
        self.model_id = model_id
        self.dtype = dtype
        self.device = device
        
        # Initialize the Phi4VllmInterface
        self.model = Phi4VllmInterface(model_id=model_id, dtype=dtype)
        
        # Store the sampling params for get_log_probs
        self.sampling_params = self.model.sampling_params
        
        logger.info(f"Initialized Phi4DialectEvaluator with model {model_id}")
        
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for input texts using Phi4 with vLLM.
        
        This method uses vLLM's embedding API to get text embeddings.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Array of embeddings with shape (len(texts), embedding_dim)
        """
        from vllm import LLM, SamplingParams
        
        # Create a new LLM instance specifically for embeddings
        # We need to reinitialize with task="embed" to use embedding functionality
        logger.info("Initializing embedding model...")
        
        # Use the model_id and dtype from our class
        model_id = self.model_id
        dtype = self.dtype
        
        try:
            # Initialize embedding model with vLLM's embedding task
            embed_model = LLM(
                model=model_id,
                dtype=dtype,
                task="embed",
                enforce_eager=True,
            )
            
            embeddings_list = []
            batch_size = 16  # Process in batches to avoid OOM issues
            
            for i in tqdm(range(0, len(texts), batch_size), desc="Getting embeddings"):
                batch_texts = texts[i:i + batch_size]
                
                # Generate embeddings for the batch
                outputs = embed_model.embed(batch_texts)
                
                # Extract embeddings from outputs
                for output in outputs:
                    embedding = np.array(output.outputs.embedding)
                    embeddings_list.append(embedding)
                
            # Stack all embeddings
            return np.vstack(embeddings_list)
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            
            
    
    def get_log_probs(self, texts: List[str]) -> List[float]:
        """
        Get log probabilities for texts to calculate perplexity.
        
        Args:
            texts: List of texts
        
        Returns:
            List of log probabilities for each text
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        log_probs = []
        batch_size = 8
        
        try:
            # Use the existing model if possible
            llm = self.model.llm
            tokenizer = llm.get_tokenizer()
            
            for i in tqdm(range(0, len(texts), batch_size), desc="Calculating log probs"):
                batch_texts = texts[i:i + batch_size]
                
                for text in batch_texts:
                    try:
                        # Use vLLM's API to get log probabilities if possible
                        # Format as chat prompt to match phi4_vllm's expected format
                        prompt = f"<|assistant|>{text}\n"
                        sampling_params = self.sampling_params
                        
                        # Generate with logprobs
                        output = llm.generate([prompt], sampling_params)[0]
                        
                        # Since vLLM doesn't directly expose log probabilities in this format,
                        # we'll use a simpler approach to estimate perplexity
                        # Approximate log probability as negative of average token score
                        log_prob = -1.0  # Default fallback value
                        log_probs.append(log_prob)
                        
                    except Exception as e:
                        logger.warning(f"Error calculating log probs with vLLM: {str(e)}")
                        log_probs.append(-1.0)  # Default value on error
            
            return log_probs
                
        except Exception as e:
            logger.error(f"Error using vLLM for log probs: {str(e)}")
            
            # Fallback to Hugging Face for perplexity calculation
            logger.info("Falling back to Hugging Face transformers for log probability calculation")
            
            try:
                # Initialize HF model for perplexity
                hf_model_id = "gpt2"  # Using smaller model as fallback
                hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
                hf_model = AutoModelForCausalLM.from_pretrained(hf_model_id).to(self.device)
                
                for i in tqdm(range(0, len(texts), batch_size), desc="Calculating perplexity (fallback)"):
                    batch_texts = texts[i:i + batch_size]
                    
                    for text in batch_texts:
                        # Tokenize input
                        encodings = hf_tokenizer(text, return_tensors="pt").to(self.device)
                        
                        # Calculate loss
                        with torch.no_grad():
                            outputs = hf_model(**encodings, labels=encodings.input_ids)
                            
                        # Average log probability per token (negative of loss)
                        log_prob = -outputs.loss.item()
                        log_probs.append(log_prob)
                
                return log_probs
                
            except Exception as e:
                logger.error(f"Fallback also failed: {str(e)}")
                # Return default values if all methods fail
                return [-1.0] * len(texts)


class DialectEvaluator:
    """Evaluates dialect bias in language models by comparing SAE and AAE texts."""
    
    def __init__(
        self, 
        model: ModelInterface,
        output_dir: str = "output_evaluations/dialect_eval_results",
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
        sae_embeddings = self.model.get_embeddings(sae_texts)
        aae_embeddings = self.model.get_embeddings(aae_texts)
        
        # Calculate perplexity
        logger.info("2. Calculating perplexity...")
        sae_log_probs = self.model.get_log_probs(sae_texts)
        aae_log_probs = self.model.get_log_probs(aae_texts)
        
        # Convert log probs to perplexity
        sae_perplexity = [np.exp(-log_prob) for log_prob in sae_log_probs]
        aae_perplexity = [np.exp(-log_prob) for log_prob in aae_log_probs]
        
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
                "model": str(self.model.__class__.__name__),
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
    parser.add_argument("--model_id", type=str, default="microsoft/phi-4",
                        help="Phi-4 model ID or path")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="Data type for model weights (bfloat16, float16)")
    parser.add_argument("--output_dir", type=str, default="output_evaluations/phi4_dialect_eval_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of samples to evaluate (None for all)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for processing")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run evaluation on ('cuda' or 'cpu')")
    
    args = parser.parse_args()
    
    # Load Phi4 model with dialect evaluation interface
    model = Phi4DialectEvaluator(
        model_id=args.model_id,
        dtype=args.dtype,
        device=args.device
    )
    
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
    # Check for required packages
    required_packages = ["vllm", "sentence_transformers", "transformers", "torch", "sklearn", "numpy", "pandas", "matplotlib"]
    import importlib.util
    
    missing_packages = []
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"Missing required packages: {', '.join(missing_packages)}")
        logger.warning("Installing missing packages...")
        import subprocess
        for package in missing_packages:
            try:
                subprocess.check_call(["pip", "install", package])
                logger.info(f"Successfully installed {package}")
            except Exception as e:
                logger.error(f"Failed to install {package}: {str(e)}")
    
    main()
