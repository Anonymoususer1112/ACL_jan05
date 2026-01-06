"""
Test script for the Phi-4 vLLM interface.
"""

import sys
import os
import logging
import traceback

# Force output to stdout
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add the parent directory to the sys.path to import the models package
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
logger.debug(f"Added {parent_dir} to sys.path")

try:
    print("Importing Phi4VllmInterface...")
    from models.phi4_vllm import Phi4VllmInterface
    print("Successfully imported Phi4VllmInterface")
except Exception as e:
    print(f"Error importing Phi4VllmInterface: {str(e)}")
    traceback.print_exc()
    sys.exit(1)

def main():
    """Test the Phi-4 vLLM interface."""
    print("Starting test of Phi-4 vLLM interface")
    
    try:
        # Initialize the interface
        print("Initializing Phi4VllmInterface...")
        phi4 = Phi4VllmInterface()
        print("Successfully initialized Phi4VllmInterface")
        
        # Test a simple prompt
        prompt = "Hello, how are you today?"
        print(f"Sending prompt: {prompt}")
        
        response = phi4.call_model(prompt)
        print(f"Response: {response}")
        
        # Test sentiment analysis
        sentiment_text = "I love this new feature, it works great!"
        print(f"Testing sentiment analysis for: {sentiment_text}")
        
        sentiment = phi4.get_sentiment(sentiment_text)
        print(f"Sentiment: {sentiment}")
        
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error testing Phi-4 vLLM interface: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
