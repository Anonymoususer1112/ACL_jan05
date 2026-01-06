"""
Claude model interface for Anthropic Batch API.
"""
import os
import sys
import time
import json
import logging
import tempfile
import backoff
from typing import Dict, List, Any, Optional

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

# Change relative import to absolute import
try:
    from .model_interface import APIModelInterface
except ImportError:
    import sys
    import os
    # Add parent directory to path
    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    from models.model_interface import APIModelInterface

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

class ClaudeBatchInterface(APIModelInterface):
    """Interface for Claude models through Anthropic Batch API."""
    
    def __init__(self, api_key: str = None, model_id: str = "claude-3-7-sonnet-20250219"):
        """Initialize the Claude Batch interface.
        
        Args:
            api_key: Anthropic API key (if not set in environment)
            model_id: Claude model version to use
        """
        super().__init__("claude_batch", api_key)
        
        # Configure client
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # Set Claude-specific parameters
        self.model_id = model_id
        self.rate_limit_per_minute = 60  # Adjust based on API tier
        
        # Configure retries and rate limits
        self.max_retries = 5
        self.retry_delay = 5
        
        # Keep track of batches
        self.active_batches = {}
        
        # Maximum batch file size (100MB safety margin)
        self.max_file_size_mb = 200  # Claude batch API can handle 256MB, but keep margin for safety
    
    @backoff.on_exception(backoff.expo, 
                         (anthropic.RateLimitError, anthropic.APIError), 
                         max_tries=5)
    def call_model(self, text: str) -> str:
        """Call the Claude model with a text prompt.
        
        Args:
            text: Input text/prompt
            
        Returns:
            Model response as string
        """
        self._check_rate_limit()
        
        try:
            response = self.client.messages.create(
                model=self.model_id,
                max_tokens=1000,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": text}
                ]
            )
            
            # Extract the response text
            if response.content and len(response.content) > 0:
                return response.content[0].text
            else:
                logger.warning("No response from Claude")
                return "ERROR: No response generated"
                
        except anthropic.RateLimitError as e:
            logger.warning(f"Rate limit exceeded: {str(e)}")
            raise  # Will be caught by backoff decorator
            
        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise  # Will be caught by backoff decorator
            
        except Exception as e:
            logger.error(f"Error calling Claude: {str(e)}")
            return f"ERROR: {str(e)}"
    
    def _check_rate_limit(self):
        """Implement simple rate limiting."""
        # Check if we need to implement rate limiting
        # For now, this is just a placeholder
        pass

    def _create_request_for_single_text(self, text: str, prompt_template: str = None, custom_id: str = None) -> Request:
        """Create a single batch request for Claude.
        
        Args:
            text: Text to process
            prompt_template: Optional template to format the text
            custom_id: Custom ID for the request
            
        Returns:
            Request object
        """
        try:
            # Use custom_id or generate a default one
            if custom_id is None:
                custom_id = f"request-{hash(text) % 10000}" # Simple hash for uniqueness
                
            # Format the text with the template if provided
            if prompt_template:
                formatted_text = prompt_template.format(text=text)
            else:
                formatted_text = text
                
            request = Request(
                custom_id=custom_id,
                params=MessageCreateParamsNonStreaming(
                    model=self.model_id,
                    max_tokens=1024,
                    messages=[{
                        "role": "user",
                        "content": formatted_text
                    }]
                )
            )
            return request
        except Exception as e:
            logger.error(f"Error creating batch request: {str(e)}")
            raise

    def submit_batch(self, requests: List[Request], description: str = None) -> str:
        """Submit a batch processing job.
        
        Args:
            requests: List of Request objects
            description: Optional description for the batch
            
        Returns:
            Batch ID
        """
        try:
            # Create the batch
            batch = self.client.messages.batches.create(requests=requests)
            
            batch_id = batch.id
            logger.info(f"Created batch with ID: {batch_id}, status: {batch.processing_status}")
            
            # Store batch info
            self.active_batches[batch_id] = {
                "id": batch_id,
                "status": batch.processing_status,
                "submitted_at": time.time()
            }
            
            return batch_id
        
        except Exception as e:
            logger.error(f"Error submitting batch: {str(e)}")
            raise

    def check_batch_status(self, batch_id: str) -> Dict:
        """Check the status of a batch.
        
        Args:
            batch_id: ID of the batch to check
            
        Returns:
            Batch status information
        """
        try:
            batch = self.client.messages.batches.retrieve(batch_id)
            
            # Update stored batch info
            if batch_id in self.active_batches:
                self.active_batches[batch_id]["status"] = batch.processing_status
                if batch.results_url:
                    self.active_batches[batch_id]["results_url"] = batch.results_url
            
            return {
                "id": batch.id,
                "status": batch.processing_status,
                "progress": {
                    "total": batch.request_counts.processing + batch.request_counts.succeeded + 
                             batch.request_counts.errored + batch.request_counts.canceled + 
                             batch.request_counts.expired,
                    "succeeded": batch.request_counts.succeeded,
                    "errored": batch.request_counts.errored,
                    "processing": batch.request_counts.processing,
                    "canceled": batch.request_counts.canceled,
                    "expired": batch.request_counts.expired
                },
                "results_url": batch.results_url,
                "created_at": batch.created_at,
                "ended_at": batch.ended_at
            }
        
        except Exception as e:
            logger.error(f"Error checking batch status: {str(e)}")
            raise

    def wait_for_batch_completion(self, batch_id: str, polling_interval: int = 60, timeout: int = 3600) -> Dict:
        """Wait for a batch to complete, with timeout.
        
        Args:
            batch_id: ID of the batch to wait for
            polling_interval: How often to check status in seconds
            timeout: Maximum time to wait in seconds
            
        Returns:
            Final batch status
        """
        start_time = time.time()
        while True:
            status = self.check_batch_status(batch_id)
            
            if status["status"] == "ended":
                logger.info(f"Batch {batch_id} finished processing")
                return status
            
            if time.time() - start_time > timeout:
                logger.warning(f"Timeout waiting for batch {batch_id} to complete")
                return status
            
            # Log progress
            progress = status["progress"]
            if progress["total"] > 0:
                percent_complete = (progress["succeeded"] + progress["errored"] + 
                                  progress["canceled"] + progress["expired"]) / progress["total"] * 100
                logger.info(f"Batch {batch_id} progress: {percent_complete:.1f}% " + 
                          f"(Processing: {progress['processing']}, Succeeded: {progress['succeeded']}, " +
                          f"Errored: {progress['errored']}, Total: {progress['total']})")
            
            time.sleep(polling_interval)

    def get_batch_results(self, batch_id: str) -> List[Dict]:
        """Get the results of a completed batch.
        
        Args:
            batch_id: ID of the batch to get results for
            
        Returns:
            List of response dictionaries
        """
        try:
            results = []
            for result in self.client.messages.batches.results(batch_id):
                results.append(result) 
            return results
        
        except Exception as e:
            logger.error(f"Error getting batch results: {str(e)}")
            raise

    def extract_responses_from_batch(self, batch_results: List[Dict]) -> Dict[str, str]:
        """Extract the actual responses from batch results.
        
        Args:
            batch_results: Raw batch results from get_batch_results
            
        Returns:
            Dictionary mapping custom_ids to response content strings
        """
        responses = {}
        for result in batch_results:
            custom_id = result.custom_id
            
            if result.result.type == "succeeded":
                # Extract message content from the successful response
                message = result.result.message
                if message and message.content and len(message.content) > 0:
                    responses[custom_id] = message.content[0].text
                else:
                    responses[custom_id] = "ERROR: No content in response"
            else:
                # For error cases, record the error type
                responses[custom_id] = f"ERROR: {result.result.type}"
                if result.result.type == "errored" and hasattr(result.result, "error"):
                    responses[custom_id] += f" - {result.result.error.type}"
        
        return responses

    def batch_get_sentiment(self, texts: List[str], batch_size: int = None) -> List[Dict[str, Any]]:
        """Get sentiment classifications for a batch of texts using Batch API.
        
        Args:
            texts: List of input texts
            batch_size: Size of each batch (not used, auto-determined based on API limits)
            
        Returns:
            List of sentiment dictionaries
        """
        if not texts:
            return []
            
        # Create a batch file for sentiment analysis
        prompt_template = """
        Please analyze the sentiment of the following text and respond with exactly one word: 
        either 'positive', 'negative', or 'neutral'.

        Text: "{text}"
        
        Sentiment:
        """
        
        try:
            # Create all requests at once
            all_requests = []
            for i, text in enumerate(texts):
                custom_id = f"request-{i}"
                # Format the text with the template
                formatted_text = prompt_template.format(text=text)
                
                request = Request(
                    custom_id=custom_id,
                    params=MessageCreateParamsNonStreaming(
                        model=self.model_id,
                        max_tokens=1024,
                        messages=[{
                            "role": "user",
                            "content": formatted_text
                        }]
                    )
                )
                all_requests.append(request)
            
            # Check if we need to split due to API limits (2000 requests per batch)
            MAX_BATCH_REQUESTS = 2000
            all_responses = {}
            
            if len(all_requests) <= MAX_BATCH_REQUESTS:
                # Submit the entire batch at once
                logger.info(f"Submitting entire batch of {len(all_requests)} requests at once")
                batch_id = self.submit_batch(all_requests, "Sentiment analysis batch")
                status = self.wait_for_batch_completion(batch_id)
                
                if status["status"] != "ended":
                    logger.error(f"Batch {batch_id} did not complete successfully: {status}")
                    # Add failed responses
                    for i in range(len(texts)):
                        all_responses[f"request-{i}"] = f"ERROR: Batch error - {status['status']}"
                else:
                    # Get results and parse responses
                    batch_results = self.get_batch_results(batch_id)
                    all_responses = self.extract_responses_from_batch(batch_results)
            else:
                # Need to split into multiple batches due to API limits
                logger.warning(f"Batch too large ({len(all_requests)} requests), splitting into multiple batches")
                for i in range(0, len(all_requests), MAX_BATCH_REQUESTS):
                    batch_slice = all_requests[i:i+MAX_BATCH_REQUESTS]
                    batch_num = i // MAX_BATCH_REQUESTS + 1
                    total_batches = (len(all_requests) + MAX_BATCH_REQUESTS - 1) // MAX_BATCH_REQUESTS
                    
                    logger.info(f"Submitting batch {batch_num}/{total_batches} with {len(batch_slice)} requests")
                    batch_id = self.submit_batch(batch_slice, f"Sentiment analysis batch {batch_num}/{total_batches}")
                    status = self.wait_for_batch_completion(batch_id)
                    
                    if status["status"] != "ended":
                        logger.error(f"Batch {batch_id} did not complete successfully: {status}")
                        # Add failed responses for this batch
                        for j in range(len(batch_slice)):
                            request_index = i + j
                            all_responses[f"request-{request_index}"] = f"ERROR: Batch error - {status['status']}"
                        continue
                    
                    # Get results and parse responses
                    batch_results = self.get_batch_results(batch_id)
                    batch_responses = self.extract_responses_from_batch(batch_results)
                    all_responses.update(batch_responses)
            
            # Process responses into sentiment dictionaries
            sentiments = []
            for i in range(len(texts)):
                custom_id = f"request-{i}"
                response = all_responses.get(custom_id, f"ERROR: No response for {custom_id}")
                
                # Clean up the response
                response = response.strip().lower()
                
                # Check if response contains one of the expected sentiments
                if 'positive' in response:
                    sentiment = 'positive'
                    score = 1
                elif 'negative' in response:
                    sentiment = 'negative'
                    score = -1
                elif 'neutral' in response:
                    sentiment = 'neutral'
                    score = 0
                else:
                    logger.warning(f"Unexpected sentiment response: {response}")
                    sentiment = 'ERROR'
                    score = 0
                
                sentiments.append({
                    'sentiment': sentiment,
                    'score': score,
                    'raw_response': response
                })
            
            return sentiments
            
        except Exception as e:
            logger.error(f"Error in batch sentiment analysis: {str(e)}")
            return [{'sentiment': 'ERROR', 'score': 0, 'raw_response': str(e)}] * len(texts)

    def get_sentiment(self, text: str) -> Dict[str, Any]:
        """Get sentiment classification using Claude.
        
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
                score = 1
            elif 'negative' in response:
                sentiment = 'negative'
                score = -1
            elif 'neutral' in response:
                sentiment = 'neutral'
                score = 0
            else:
                logger.warning(f"Unexpected sentiment response: {response}")
                sentiment = 'ERROR'
                score = 0
            
            return {
                'sentiment': sentiment,
                'score': score,
                'raw_response': response
            }
        except Exception as e:
            logger.error(f"Error getting sentiment: {str(e)}")
            return {'sentiment': 'ERROR', 'score': 0, 'raw_response': str(e)}

    async def batch_translate_texts(self, texts: List[str], translation_prompt_template: str, description: str = "Translation batch") -> List[str]:
        """Translate a batch of texts using the Batch API.
        
        Args:
            texts: List of texts to translate
            translation_prompt_template: The prompt template for translation with {text} placeholder
            description: Description for the batch job
            
        Returns:
            List of translated texts
        """
        if not texts:
            return []
            
        try:
            # Create all requests at once
            all_requests = []
            for i, text in enumerate(texts):
                custom_id = f"request-{i}"
                # Format the text with the template
                formatted_text = translation_prompt_template.format(text=text)
                
                request = Request(
                    custom_id=custom_id,
                    params=MessageCreateParamsNonStreaming(
                        model=self.model_id,
                        max_tokens=1024,
                        messages=[{
                            "role": "user",
                            "content": formatted_text
                        }]
                    )
                )
                all_requests.append(request)
            
            # Check if we need to split due to API limits (2000 requests per batch)
            MAX_BATCH_REQUESTS = 2000
            all_responses = {}
            
            if len(all_requests) <= MAX_BATCH_REQUESTS:
                # Submit the entire batch at once
                logger.info(f"Submitting entire translation batch of {len(all_requests)} requests at once")
                batch_id = self.submit_batch(all_requests, description)
                status = self.wait_for_batch_completion(batch_id)
                
                if status["status"] != "ended":
                    logger.error(f"Batch {batch_id} did not complete successfully: {status}")
                    # Add failed responses
                    for i in range(len(texts)):
                        all_responses[f"request-{i}"] = f"TRANSLATION_FAILED: Batch error - {status['status']}"
                else:
                    # Get results and parse responses
                    batch_results = self.get_batch_results(batch_id)
                    all_responses = self.extract_responses_from_batch(batch_results)
            else:
                # Need to split into multiple batches due to API limits
                logger.warning(f"Translation batch too large ({len(all_requests)} requests), splitting into multiple batches")
                for i in range(0, len(all_requests), MAX_BATCH_REQUESTS):
                    batch_slice = all_requests[i:i+MAX_BATCH_REQUESTS]
                    batch_num = i // MAX_BATCH_REQUESTS + 1
                    total_batches = (len(all_requests) + MAX_BATCH_REQUESTS - 1) // MAX_BATCH_REQUESTS
                    
                    logger.info(f"Submitting translation batch {batch_num}/{total_batches} with {len(batch_slice)} requests")
                    batch_id = self.submit_batch(batch_slice, f"{description} {batch_num}/{total_batches}")
                    status = self.wait_for_batch_completion(batch_id)
                    
                    if status["status"] != "ended":
                        logger.error(f"Batch {batch_id} did not complete successfully: {status}")
                        # Add failed responses for this batch
                        for j in range(len(batch_slice)):
                            request_index = i + j
                            all_responses[f"request-{request_index}"] = f"TRANSLATION_FAILED: Batch error - {status['status']}"
                        continue
                    
                    # Get results and parse responses
                    batch_results = self.get_batch_results(batch_id)
                    batch_responses = self.extract_responses_from_batch(batch_results)
                    all_responses.update(batch_responses)
            
            # Process the responses to extract translations
            translations = []
            for i in range(len(texts)):
                custom_id = f"request-{i}"
                response = all_responses.get(custom_id, f"TRANSLATION_FAILED: No response for {custom_id}")
                translations.append(response)
            
            return translations
            
        except Exception as e:
            logger.error(f"Error in batch translation: {str(e)}")
            return [f"TRANSLATION_FAILED: {str(e)}"] * len(texts)

    def translate_aae_to_sae(self, text: str) -> str:
        """Translate an AAE text to SAE.
        
        Args:
            text: AAE text to translate
            
        Returns:
            Translated SAE text
        """
        prompt = f"""
        Translate the following tweet from African American English (AAE) to Standard American English (SAE).
        Preserve the meaning, tone, and intent of the original tweet.
        Only change dialectical features while maintaining the original message.
        Original tweet (AAE): "{text}"
        Standard American English translation:
        """
        
        try:
            response = self.call_model(prompt)
            
            # Extract translation from response
            if "Standard American English translation:" in response:
                response = response.split("Standard American English translation:")[1].strip()
            
            # If no specific format found, return the whole response
            return response.strip()
        except Exception as e:
            logger.error(f"Error translating AAE to SAE: {str(e)}")
            return f"TRANSLATION_FAILED: {text}"

    def translate_sae_to_aae(self, text: str) -> str:
        """Translate an SAE text to AAE.
        
        Args:
            text: SAE text to translate
            
        Returns:
            Translated AAE text
        """
        prompt = f"""
        Translate the following tweet from Standard American English (SAE) to African American English (AAE).
        Preserve the meaning, tone, and intent of the original tweet.
        Only change dialectical features while maintaining the original message.
        Original tweet (SAE): "{text}"
        African American English translation:
        """
        
        try:
            response = self.call_model(prompt)
            
            # Extract translation from response
            if "African American English translation:" in response:
                response = response.split("African American English translation:")[1].strip()
            
            # If no specific format found, return the whole response
            return response.strip()
        except Exception as e:
            logger.error(f"Error translating SAE to AAE: {str(e)}")
            return f"TRANSLATION_FAILED: {text}"

def call_claude(text: str, model_id: str = "claude-3-5-sonnet-20240620") -> str:
    """Function to call Claude model.
    
    Args:
        text: Input text/prompt
        model_id: Claude model to use
        
    Returns:
        Model response
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    model = ClaudeBatchInterface(api_key=api_key, model_id=model_id)
    return model.call_model(text)
