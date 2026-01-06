"""
Base model interface for dialect evaluation.
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class ModelInterface(ABC):
    """
    Interface for models used in dialect evaluation.
    Implement this for any model you want to evaluate.
    """
    
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for input texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Array of embeddings with shape (len(texts), embedding_dim)
        """
        pass
    
    @abstractmethod
    def get_log_probs(self, texts: List[str]) -> List[float]:
        """
        Get average log probabilities for input texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of average log probabilities (negative values)
        """
        pass


class HuggingFaceModelInterface(ModelInterface):
    """Example implementation for Hugging Face models."""
    
    def __init__(self, model_name_or_path, device="cpu"):
        """Initialize with a Hugging Face model."""
        from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
        import torch
        
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        # Load embedding model
        self.embedding_model = AutoModel.from_pretrained(model_name_or_path).to(device)
        
        # Load LM for perplexity
        self.lm_model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from Hugging Face model."""
        import torch
        
        # Tokenize
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt", 
            max_length=512
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
        
        # Mean pooling
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = (sum_embeddings / sum_mask).cpu().numpy()
        
        return embeddings
    
    def get_log_probs(self, texts: List[str]) -> List[float]:
        """Get log probabilities from Hugging Face model."""
        import torch
        
        log_probs = []
        
        for text in texts:
            # Tokenize input
            encodings = self.tokenizer(text, return_tensors="pt").to(self.device)
            
            # Get sequence length (excluding special tokens)
            seq_len = encodings.input_ids.size(1)
            
            # Calculate log probability
            with torch.no_grad():
                outputs = self.lm_model(**encodings, labels=encodings.input_ids)
                loss = outputs.loss.item()
                
            # Average log probability per token (negative of loss)
            log_prob = -loss
            log_probs.append(log_prob)
        
        return log_probs


# Example model loader function
def load_model(model_path_or_name, device="cpu"):
    """
    Load a model by name or path.
    
    Args:
        model_path_or_name: Path to model or model identifier
        device: Device to load model on
        
    Returns:
        Loaded model with ModelInterface
    """
    # For this example we're just using the HuggingFace interface
    return HuggingFaceModelInterface(model_path_or_name, device)