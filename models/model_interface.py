"""
Model interface for LLM API calls and sentiment classification.

This module provides base classes and utilities for calling various LLMs
for sentiment classification and dialect translation tasks.
"""

import os
import asyncio
import logging
import time
import json
import numpy as np
from typing import List, Dict, Union, Optional, Any, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelInterface:
    """Base class for model interfaces."""
    
    def __init__(self, model_name: str):
        """Initialize the model interface.
        
        Args:
            model_name: Name of the model
        """
        self.model_name = model_name
        logger.info(f"Initializing {model_name} interface")

    def call_model(self, text: str) -> str:
        """Call the model with the given text.
        
        Args:
            text: Input text to the model
            
        Returns:
            Model response as string
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def batch_call_model(self, texts: List[str], batch_size: int = 8) -> List[str]:
        """Call the model with a batch of texts.
        
        Args:
            texts: List of input texts
            batch_size: Size of each batch
            
        Returns:
            List of model responses
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                batch_results = [self.call_model(text) for text in batch]
                results.extend(batch_results)
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                # Mark failed results and continue
                failed_results = ["ERROR: Processing failed"] * len(batch)
                results.extend(failed_results)
            
            # Add delay to avoid rate limiting
            time.sleep(1)
        
        return results
    
    async def async_call_model(self, text: str) -> str:
        """Async version of call_model.
        
        Args:
            text: Input text to the model
            
        Returns:
            Model response as string
        """
        # Default implementation just calls the sync version
        # Subclasses should implement a proper async version if supported
        return self.call_model(text)
    
    async def async_batch_call_model(self, texts: List[str], 
                                    batch_size: int = 10,
                                    max_concurrency: int = 5) -> List[str]:
        """Call the model with a batch of texts asynchronously.
        
        Args:
            texts: List of input texts
            batch_size: Size of each batch for logging purposes
            max_concurrency: Maximum number of concurrent requests
            
        Returns:
            List of model responses
        """
        results = [""] * len(texts)
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def process_item(i, text):
            async with semaphore:
                try:
                    results[i] = await self.async_call_model(text)
                    if (i + 1) % 10 == 0:
                        logger.info(f"Processed {i + 1}/{len(texts)} items")
                except Exception as e:
                    logger.error(f"Error processing item {i}: {str(e)}")
                    results[i] = "ERROR: Processing failed"
                
                # Add small delay to avoid rate limiting
                await asyncio.sleep(0.1)
        
        # Create tasks for all texts
        tasks = [process_item(i, text) for i, text in enumerate(texts)]
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
        return results

    def get_sentiment(self, text: str) -> Dict[str, Any]:
        """Get sentiment classification for a text.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with sentiment classification: {'sentiment': str, 'score': float}
            where sentiment is one of 'positive', 'negative', 'neutral'
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def batch_get_sentiment(self, texts: List[str], batch_size: int = 16) -> List[Dict[str, Any]]:
        """Get sentiment classifications for a batch of texts.
        
        Args:
            texts: List of input texts
            batch_size: Size of each batch
            
        Returns:
            List of sentiment dictionaries
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                batch_results = [self.get_sentiment(text) for text in batch]
                results.extend(batch_results)
                logger.info(f"Processed sentiment batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            except Exception as e:
                logger.error(f"Error processing sentiment batch {i//batch_size + 1}: {str(e)}")
                # Mark failed results and continue
                failed_results = [{"sentiment": "ERROR", "score": 0.0}] * len(batch)
                results.extend(failed_results)
            
            # Add delay to avoid overloading
            time.sleep(0.5)
        
        return results

class TransformersModelInterface(ModelInterface):
    """Interface for HuggingFace Transformers models."""
    
    def __init__(self, model_name: str, model_path: str):
        """Initialize the model interface.
        
        Args:
            model_name: Name of the model
            model_path: Path or identifier for the HuggingFace model
        """
        super().__init__(model_name)
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model and tokenizer."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path).to(self.device)
            logger.info(f"Successfully loaded model {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def call_model(self, text: str) -> str:
        """Call the model with the given text.
        
        This generic implementation might not be suitable for all models.
        Subclasses should override this method as needed.
        
        Args:
            text: Input text to the model
            
        Returns:
            Model response as string
        """
        try:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error calling model: {str(e)}")
            return f"ERROR: {str(e)}"

    def batch_call_model(self, texts: List[str], batch_size: int = 8) -> List[str]:
        """Efficient batch processing for transformer models.
        
        Args:
            texts: List of input texts
            batch_size: Size of each batch
            
        Returns:
            List of model responses
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                # Tokenize all texts in the batch at once
                inputs = self.tokenizer(batch, padding=True, truncation=True, 
                                      return_tensors="pt").to(self.device)
                
                # Process batch
                with torch.no_grad():
                    outputs = self.model.generate(**inputs)
                
                # Decode all outputs
                batch_results = [self.tokenizer.decode(output, skip_special_tokens=True) 
                               for output in outputs]
                results.extend(batch_results)
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                failed_results = ["ERROR: Processing failed"] * len(batch)
                results.extend(failed_results)
        
        return results
    
    def get_sentiment(self, text: str) -> Dict[str, Any]:
        """Get sentiment classification using a transformer model.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with sentiment classification
        """
        try:
            # Create sentiment analysis pipeline if not already done
            if not hasattr(self, 'sentiment_pipeline'):
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis", 
                    model=self.model, 
                    tokenizer=self.tokenizer,
                    # device=0 if self.device == "cuda" else -1
                )
            
            # Get sentiment
            result = self.sentiment_pipeline(text)[0]
            
            # Map to standard format (positive, negative, neutral)
            label = result['label'].lower()
            if 'positive' in label:
                sentiment = 'positive'
            elif 'negative' in label:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
                
            return {
                'sentiment': sentiment,
                'score': result['score'],
                'original_label': result['label']
            }
        except Exception as e:
            logger.error(f"Error getting sentiment: {str(e)}")
            return {'sentiment': 'ERROR', 'score': 0.0}
    
    def batch_get_sentiment(self, texts: List[str], batch_size: int = 16) -> List[Dict[str, Any]]:
        """Get sentiment classifications for a batch of texts.
        
        Args:
            texts: List of input texts
            batch_size: Size of each batch
            
        Returns:
            List of sentiment dictionaries
        """
        # Create sentiment analysis pipeline if not already done
        if not hasattr(self, 'sentiment_pipeline'):
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis", 
                model=self.model, 
                tokenizer=self.tokenizer,
                # device=0 if self.device == "cuda" else -1
            )
        
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                # Process batch
                batch_results = self.sentiment_pipeline(batch)
                
                # Map to standard format
                processed_results = []
                for result in batch_results:
                    label = result['label'].lower()
                    if 'positive' in label:
                        sentiment = 'positive'
                    elif 'negative' in label:
                        sentiment = 'negative'
                    else:
                        sentiment = 'neutral'
                        
                    processed_results.append({
                        'sentiment': sentiment,
                        'score': result['score'],
                        'original_label': result['label']
                    })
                
                results.extend(processed_results)
                logger.info(f"Processed sentiment batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            except Exception as e:
                logger.error(f"Error processing sentiment batch {i//batch_size + 1}: {str(e)}")
                failed_results = [{"sentiment": "ERROR", "score": 0.0}] * len(batch)
                results.extend(failed_results)
        
        return results

class APIModelInterface(ModelInterface):
    """Interface for API-based models like GPT or Claude."""
    
    def __init__(self, model_name: str, api_key: str = None):
        """Initialize the API model interface.
        
        Args:
            model_name: Name of the model
            api_key: API key (if None, will try to get from environment variables)
        """
        super().__init__(model_name)
        self.api_key = api_key or os.environ.get(f"{model_name.upper()}_API_KEY")
        if not self.api_key:
            logger.warning(f"No API key provided for {model_name}. Set the {model_name.upper()}_API_KEY environment variable.")
        
        # API specific configurations should be set in subclasses
        self.api_url = None
        self.headers = {}
        
        # Set up rate limiting parameters
        self.request_count = 0
        self.last_request_time = 0
        self.rate_limit_per_minute = 60  # Default, can be overridden in subclasses

    def _check_rate_limit(self):
        """Check and enforce rate limits."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        # If less than a minute has passed since first request in current window
        if elapsed < 60 and self.request_count >= self.rate_limit_per_minute:
            # Sleep for the remaining time to complete a minute
            sleep_time = 60 - elapsed
            logger.info(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
            
            # Reset counters
            self.request_count = 0
            self.last_request_time = time.time()
        
        # If more than a minute has passed, reset counters
        elif elapsed >= 60:
            self.request_count = 0
            self.last_request_time = current_time
        
        # Increment request counter
        self.request_count += 1
    
    def call_model(self, text: str) -> str:
        """Call the API with the given text.
        
        Args:
            text: Input text to the model
            
        Returns:
            Model response as string
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    async def async_call_model(self, text: str) -> str:
        """Async version of call_model.
        
        Args:
            text: Input text to the model
            
        Returns:
            Model response as string
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_sentiment(self, text: str) -> Dict[str, Any]:
        """Get sentiment classification using the API.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with sentiment classification
        """
        prompt = f"""
        Please analyze the sentiment of the following text and respond with exactly one word: 
        either 'positive', 'negative', or 'neutral'.

        Text: "{text}"
        
        Sentiment:
        """
        
        try:
            response = self.call_model(prompt)
            response = response.strip().lower()
            
            # Check if response contains one of the expected sentiments
            if 'positive' in response:
                sentiment = 'positive'
            elif 'negative' in response:
                sentiment = 'negative'
            elif 'neutral' in response:
                sentiment = 'neutral'
            else:
                logger.warning(f"Unexpected sentiment response: {response}")
                sentiment = 'ERROR'
            
            return {
                'sentiment': sentiment,
                'raw_response': response
            }
        except Exception as e:
            logger.error(f"Error getting sentiment: {str(e)}")
            return {'sentiment': 'ERROR', 'raw_response': str(e)}

# Function to map sentiment values to numeric scores
def sentiment_to_score(sentiment: str) -> int:
    """Convert sentiment string to numeric score.
    
    Args:
        sentiment: One of 'positive', 'negative', 'neutral'
        
    Returns:
        1 for positive, -1 for negative, 0 for neutral
    """
    sentiment = sentiment.lower()
    if 'positive' in sentiment:
        return 1
    elif 'negative' in sentiment:
        return -1
    else:
        return 0