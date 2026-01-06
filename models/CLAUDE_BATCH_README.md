# Claude Batch Interface for Sentiment Analysis

This document provides a guide on using the Claude Batch API interface for sentiment analysis on large datasets.

## Overview

The Claude Batch Interface (`ClaudeBatchInterface`) is designed to efficiently process large datasets using Anthropic's Claude AI models through their batch API. This allows you to analyze sentiment for thousands of texts in an asynchronous, efficient manner.

## Key Features

- **Batch Processing**: Process large datasets efficiently using Anthropic's batch API
- **Automatic Batching**: Automatically splits large datasets into appropriate batch sizes
- **Resilient Error Handling**: Gracefully handles errors and provides detailed logging
- **Sentiment Analysis**: Built-in methods for sentiment classification
- **Translation Support**: Methods for translating between AAE and SAE dialects
- **Polling and Status Tracking**: Monitor batch processing progress

## Setup

### Prerequisites

1. An Anthropic API key with access to the batch API
2. Python 3.8 or later
3. Required packages: `anthropic`, `pandas`, `backoff`, `dotenv`

### Installation

1. Ensure you have the required packages:
   ```
   pip install anthropic pandas backoff python-dotenv
   ```

2. Set your Anthropic API key:
   ```
   export ANTHROPIC_API_KEY="your_api_key_here"
   ```
   
   Alternatively, create a `.env` file with:
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ```

## Usage Examples

### Basic Sentiment Analysis

```python
from models import ClaudeBatchInterface

# Initialize the interface
claude = ClaudeBatchInterface()

# Single text sentiment analysis
result = claude.get_sentiment("I really enjoyed the movie!")
print(f"Sentiment: {result['sentiment']}, Score: {result['score']}")

# Batch sentiment analysis
texts = [
    "I love this product!",
    "I'm disappointed with the service.",
    "The weather is nice today."
]
batch_results = claude.batch_get_sentiment(texts)
```

### Processing a Dataset

```python
import pandas as pd
from models import ClaudeBatchInterface

# Load dataset
df = pd.read_csv("your_dataset.csv")

# Extract texts
texts = df["text_column"].tolist()

# Initialize interface
claude = ClaudeBatchInterface()

# Analyze sentiment
sentiments = claude.batch_get_sentiment(texts)

# Add results back to dataframe
df["sentiment"] = [s["sentiment"] for s in sentiments]
df["sentiment_score"] = [s["score"] for s in sentiments]

# Save results
df.to_csv("results.csv", index=False)
```

### Using the Command-Line Script

```bash
# Process all data in a dataset
python obtain_sentiment_scripts/claude_batch_sentiment.py \
  --input your_dataset.csv \
  --output results.csv \
  --text-column tweet_text

# Process a sample of records
python obtain_sentiment_scripts/claude_batch_sentiment.py \
  --input your_dataset.csv \
  --output sample_results.csv \
  --text-column tweet_text \
  --samples 100
```

### Using the get_aae_sentiment.py Script

```bash
# Process AAE tweets using Claude batch
python obtain_sentiment_scripts/get_aae_sentiment.py \
  --input dataset.csv \
  --output sentiment_results.csv \
  --model claude_batch \
  --text-column text \
  --batch-size 500
```

## API Reference

### ClaudeBatchInterface

```python
ClaudeBatchInterface(api_key=None, model_id="claude-3-5-sonnet-20240620")
```

- **api_key**: Anthropic API key (defaults to ANTHROPIC_API_KEY environment variable)
- **model_id**: Claude model version to use

#### Key Methods

- **call_model(text)**: Direct call to Claude for a single prompt
- **batch_get_sentiment(texts)**: Analyze sentiment for a batch of texts
- **get_sentiment(text)**: Analyze sentiment for a single text
- **submit_batch(requests, description)**: Submit a batch job
- **check_batch_status(batch_id)**: Check status of a batch job
- **wait_for_batch_completion(batch_id)**: Wait for a batch to complete
- **get_batch_results(batch_id)**: Get results from a completed batch
- **translate_aae_to_sae(text)**: Translate AAE text to SAE
- **translate_sae_to_aae(text)**: Translate SAE text to AAE
- **batch_translate_texts(texts, translation_prompt_template)**: Batch translate texts

## Troubleshooting

### Common Issues

1. **API Key Issues**:
   - Make sure your API key is correctly set in the environment or passed to the constructor
   - Check that your API key has access to the batch API

2. **Batch Processing Errors**:
   - Check Claude Batch API status
   - Ensure batch requests are valid and within size limits
   - Review error messages in the logs

3. **Rate Limiting**:
   - The interface implements backoff retries for rate limiting
   - Consider reducing batch size if rate limiting persists

### Logging

The interface produces detailed logs to help diagnose issues:

```python
import logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
```

## Best Practices

1. **Batch Size Management**:
   - The interface automatically manages batch sizes, but consider starting with smaller datasets to test
   - Monitor memory usage for very large datasets

2. **Error Handling**:
   - Check for 'ERROR' sentiment values in the output
   - Look at raw_response field for error details

3. **Monitoring**:
   - For large batches, use the check_batch_status method to monitor progress
   - Set appropriate timeout values for wait_for_batch_completion

## Advanced Features

### Custom Prompting

You can customize the sentiment analysis prompt by modifying the template:

```python
custom_prompt = """
Rate the sentiment of this text as 'very positive', 'positive', 'neutral', 'negative', or 'very negative'.

Text: "{text}"

Sentiment:
"""

# Create batch requests with custom prompt
requests = claude.create_batch_requests(texts, custom_prompt)
batch_id = claude.submit_batch(requests)
```

### Sentiment Score Normalization

The default scoring is:
- positive: 1
- neutral: 0
- negative: -1

You can customize this scoring system by modifying the return values in the batch_get_sentiment method.
