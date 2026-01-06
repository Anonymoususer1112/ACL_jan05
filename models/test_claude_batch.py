#!/usr/bin/env python3
"""
Test script for Claude Batch API interface.
"""
import os
import sys
import logging
import argparse
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import ClaudeBatchInterface

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_simple_call():
    """Test a simple direct model call."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    model = ClaudeBatchInterface(api_key=api_key)
    
    prompt = "What is the capital of France? Answer in one word."
    result = model.call_model(prompt)
    
    logger.info(f"Simple call result: {result}")
    return result

def test_batch_sentiment_analysis(num_samples: int = 10):
    """Test batch sentiment analysis on a small sample of texts."""
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
    
    # Use only the requested number of samples
    texts = texts[:num_samples]
    
    # Run batch sentiment analysis
    logger.info(f"Starting batch sentiment analysis for {len(texts)} texts")
    results = model.batch_get_sentiment(texts)
    
    # Display results
    for i, (text, result) in enumerate(zip(texts, results)):
        logger.info(f"Text {i+1}: {text[:30]}...")
        logger.info(f"Sentiment: {result['sentiment']}, Score: {result['score']}")
        logger.info(f"Raw response: {result['raw_response']}")
        logger.info("-" * 50)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Test Claude Batch API interface")
    parser.add_argument("--test-type", "-t", choices=["simple", "batch"], default="batch",
                       help="Type of test to run (simple call or batch sentiment)")
    parser.add_argument("--num-samples", "-n", type=int, default=5,
                       help="Number of samples for batch test (max 10)")
    args = parser.parse_args()
    
    if args.test_type == "simple":
        test_simple_call()
    else:
        test_batch_sentiment_analysis(min(args.num_samples, 10))

if __name__ == "__main__":
    main()
