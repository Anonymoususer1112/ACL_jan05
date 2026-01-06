#!/usr/bin/env python3
"""
Test script for updated Claude Batch API interface that sends entire batches at once.
"""
import os
import sys
import logging
import argparse
import time
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import ClaudeBatchInterface

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_batch_sentiment_analysis(num_samples: int = 10):
    """Test batch sentiment analysis with updated implementation."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    model = ClaudeBatchInterface(api_key=api_key)
    
    # Sample texts with varying sentiments
    texts = [
        "I absolutely love this new phone, it's amazing!",
        "The service at this restaurant was terrible and the food was cold.",
        "The weather is nice today, not too hot or cold.",
        "I'm so excited about the upcoming vacation!",
        "This movie was a complete waste of time and money.",
        "The report was thorough and detailed, covering all key aspects.",
        "I'm disappointed with how the project turned out.",
        "The cake was neither great nor bad, just average.",
        "I'm thrilled with my new car, it drives perfectly!",
        "The book was boring and I couldn't finish it."
    ]
    
    # Use more texts to test batch processing
    if num_samples > 10:
        # Duplicate the texts to reach the requested number
        texts = texts * (num_samples // 10 + 1)
        texts = texts[:num_samples]
    else:
        # Use only the requested number of samples
        texts = texts[:num_samples]
    
    # Run batch sentiment analysis
    start_time = time.time()
    logger.info(f"Starting batch sentiment analysis for {len(texts)} texts")
    results = model.batch_get_sentiment(texts)
    end_time = time.time()
    
    # Display results
    logger.info(f"Batch processing completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Average time per text: {(end_time - start_time) / len(texts):.4f} seconds")
    
    # Show distribution of sentiments
    sentiments = [r['sentiment'] for r in results]
    sentiment_counts = {}
    for sentiment in sentiments:
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
    
    logger.info(f"Sentiment distribution: {sentiment_counts}")
    
    # Display a few sample results
    logger.info("Sample results:")
    for i in range(min(5, len(results))):
        logger.info(f"Text: {texts[i][:30]}...")
        logger.info(f"Sentiment: {results[i]['sentiment']}, Score: {results[i]['score']}")
        logger.info(f"Raw response: {results[i]['raw_response']}")
        logger.info("-" * 50)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Test updated Claude Batch API interface")
    parser.add_argument("--num-samples", "-n", type=int, default=20,
                       help="Number of samples for batch test")
    args = parser.parse_args()
    
    test_batch_sentiment_analysis(args.num_samples)

if __name__ == "__main__":
    main()
