"""
GPT-4.1 model interface for OpenAI Batch API.
"""

import os
import json
import logging
import uuid
import time
import backoff
from typing import Dict, List, Any, Optional
import asyncio
import tempfile

import openai

from .model_interface import APIModelInterface

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

class GPT41BatchInterface(APIModelInterface):
    """Interface for GPT-4.1 through OpenAI Batch API."""
    
    def __init__(self, api_key: str = None):
        """Initialize the GPT-4.1 interface."""
        super().__init__("gpt41_batch", api_key)
        
        # Configure client
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Set GPT-4.1 specific parameters
        self.model_id = "gpt-4.1-2025-04-14"
        self.rate_limit_per_minute = 60  # Adjust based on API tier
        
        # Configure retries and rate limits
        self.max_retries = 5
        self.retry_delay = 5
        
        # Keep track of batches
        self.active_batches = {}
        
        # Maximum batch file size (OpenAI limit is 100MB)
        self.max_file_size_mb = 95  # Keeping a small safety margin
    
    @backoff.on_exception(backoff.expo, 
                         (openai.RateLimitError, openai.APIError), 
                         max_tries=5)
    def call_model(self, text: str) -> str:
        """Call the GPT-4.1 model with a text prompt.
        
        Args:
            text: Input text/prompt
            
        Returns:
            Model response as string
        """
        self._check_rate_limit()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": text}
                ],
                max_tokens=1000,
                temperature=0.3  # Lower temperature for more consistent outputs
            )
            
            # Extract the response text
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                logger.warning("No response from GPT-4.1")
                return "ERROR: No response generated"
                
        except openai.RateLimitError as e:
            logger.warning(f"Rate limit exceeded: {str(e)}")
            raise  # Will be caught by backoff decorator
            
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise  # Will be caught by backoff decorator
            
        except Exception as e:
            logger.error(f"Error calling GPT-4.1: {str(e)}")
            return f"ERROR: {str(e)}"
    
    def _check_rate_limit(self):
        """Implement simple rate limiting."""
        # Check if we need to implement rate limiting
        # For now, this is just a placeholder
        pass

    def create_batch_request_file(self, texts: List[str], prompt_template: str = None) -> tuple:
        """Create a JSONL file for batch processing.
        
        Args:
            texts: List of texts to process
            prompt_template: Optional template to format each text
            
        Returns:
            Tuple of (file path, file size in MB)
        """
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        try:
            with os.fdopen(fd, 'w') as f:
                for i, text in enumerate(texts):
                    custom_id = f"request-{i}"
                    
                    # Format the text with the template if provided
                    if prompt_template:
                        formatted_text = prompt_template.format(text=text)
                    else:
                        formatted_text = text
                        
                    request = {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": self.model_id,
                            "messages": [
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": formatted_text}
                            ],
                            "max_tokens": 1000,
                            "temperature": 0.3
                        }
                    }
                    f.write(json.dumps(request) + '\n')
            
            # Get file size in MB
            file_size_mb = os.path.getsize(path) / (1024 * 1024)
            return path, file_size_mb
        except Exception as e:
            logger.error(f"Error creating batch request file: {str(e)}")
            os.unlink(path)
            raise

    def check_file_size_and_split(self, texts: List[str], prompt_template: str) -> List[Dict[str, Any]]:
        """Check file size and determine if we need to split the batch.
        
        Args:
            texts: List of texts to process
            prompt_template: Template to format each text
            
        Returns:
            List of batch configurations, each with texts and file path
        """
        # First, try with all texts to check the size
        try:
            test_file_path, file_size_mb = self.create_batch_request_file(texts, prompt_template)
            
            # If file size is within limits, use a single batch
            if file_size_mb <= self.max_file_size_mb:
                logger.info(f"Using single batch for all {len(texts)} texts. File size: {file_size_mb:.2f} MB")
                return [{
                    "texts": texts,
                    "file_path": test_file_path,
                    "size_mb": file_size_mb
                }]
            
            # If file is too large, delete it and split into batches
            os.unlink(test_file_path)
            
            # Calculate how many batches we need
            num_batches = int((file_size_mb / self.max_file_size_mb) + 1)
            batch_size = len(texts) // num_batches
            if batch_size < 1:
                batch_size = 1
                
            logger.info(f"Splitting {len(texts)} texts into {num_batches} batches of ~{batch_size} texts each")
            
            # Create batch configurations
            batches = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_file_path, batch_size_mb = self.create_batch_request_file(batch_texts, prompt_template)
                batches.append({
                    "texts": batch_texts,
                    "file_path": batch_file_path,
                    "size_mb": batch_size_mb
                })
                
            return batches
            
        except Exception as e:
            logger.error(f"Error checking file size: {str(e)}")
            if 'test_file_path' in locals():
                try:
                    os.unlink(test_file_path)
                except:
                    pass
            raise

    def submit_batch(self, jsonl_file_path: str, description: str = None) -> str:
        """Submit a batch processing job.
        
        Args:
            jsonl_file_path: Path to the JSONL file with requests
            description: Optional description for the batch
            
        Returns:
            Batch ID
        """
        try:
            # Upload the file
            with open(jsonl_file_path, 'rb') as file:
                batch_input_file = self.client.files.create(
                    file=file,
                    purpose="batch"
                )
            
            logger.info(f"Uploaded batch file with ID: {batch_input_file.id}")
            
            # Create the batch
            batch = self.client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "description": description or "Dialectic preference analysis batch"
                }
            )
            
            batch_id = batch.id
            logger.info(f"Created batch with ID: {batch_id}, status: {batch.status}")
            
            # Store batch info
            self.active_batches[batch_id] = {
                "id": batch_id,
                "input_file_id": batch_input_file.id,
                "status": batch.status,
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
            batch = self.client.batches.retrieve(batch_id)
            
            # Update stored batch info
            if batch_id in self.active_batches:
                self.active_batches[batch_id]["status"] = batch.status
                if batch.output_file_id:
                    self.active_batches[batch_id]["output_file_id"] = batch.output_file_id
                if batch.error_file_id:
                    self.active_batches[batch_id]["error_file_id"] = batch.error_file_id
            
            return {
                "id": batch.id,
                "status": batch.status,
                "progress": {
                    "total": batch.request_counts.total,
                    "completed": batch.request_counts.completed,
                    "failed": batch.request_counts.failed
                },
                "output_file_id": batch.output_file_id,
                "error_file_id": batch.error_file_id,
                "created_at": batch.created_at,
                "completed_at": batch.completed_at
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
            
            if status["status"] in ["completed", "failed", "expired", "cancelled"]:
                logger.info(f"Batch {batch_id} finished with status: {status['status']}")
                return status
            
            if time.time() - start_time > timeout:
                logger.warning(f"Timeout waiting for batch {batch_id} to complete")
                return status
            
            # Log progress
            progress = status["progress"]
            if progress["total"] > 0:
                percent_complete = (progress["completed"] + progress["failed"]) / progress["total"] * 100
                logger.info(f"Batch {batch_id} progress: {percent_complete:.1f}% ({progress['completed']}/{progress['total']} completed)")
            
            time.sleep(polling_interval)

    def get_batch_results(self, batch_id: str) -> List[Dict]:
        """Get the results of a completed batch.
        
        Args:
            batch_id: ID of the batch to get results for
            
        Returns:
            List of response dictionaries
        """
        try:
            # Check if we have the batch info
            if batch_id not in self.active_batches:
                # Try to retrieve it
                batch = self.client.batches.retrieve(batch_id)
                if not batch.output_file_id:
                    raise ValueError(f"Batch {batch_id} has no output file")
                output_file_id = batch.output_file_id
            else:
                batch_info = self.active_batches[batch_id]
                if "output_file_id" not in batch_info:
                    # Need to check status to get output file ID
                    status = self.check_batch_status(batch_id)
                    if not status["output_file_id"]:
                        raise ValueError(f"Batch {batch_id} has no output file yet")
                    output_file_id = status["output_file_id"]
                else:
                    output_file_id = batch_info["output_file_id"]
            
            # Get the output file
            file_response = self.client.files.content(output_file_id)
            output_content = file_response.text
            
            # Parse the JSONL content
            results = []
            for line in output_content.strip().split('\n'):
                results.append(json.loads(line))
            
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
            custom_id = result.get("custom_id")
            if not custom_id:
                logger.warning(f"Result missing custom_id: {result}")
                continue
                
            response_data = result.get("response", {})
            if response_data.get("status_code") != 200:
                logger.warning(f"Error in response for {custom_id}: {response_data}")
                responses[custom_id] = f"ERROR: API returned {response_data.get('status_code')}"
                continue
                
            body = response_data.get("body", {})
            choices = body.get("choices", [])
            if not choices:
                logger.warning(f"No choices in response for {custom_id}")
                responses[custom_id] = "ERROR: No response generated"
                continue
                
            message = choices[0].get("message", {})
            content = message.get("content", "")
            responses[custom_id] = content
        
        return responses

    def batch_get_sentiment(self, texts: List[str], batch_size: int = None) -> List[Dict[str, Any]]:
        """Get sentiment classifications for a batch of texts using Batch API.
        
        Args:
            texts: List of input texts
            batch_size: Size of each batch (now optional, will auto-determine based on file size)
            
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
            # Determine if we need to split the batch based on file size
            batches = self.check_file_size_and_split(texts, prompt_template)
            
            # Process all batches
            all_responses = {}
            
            for idx, batch_config in enumerate(batches):
                logger.info(f"Processing batch {idx+1}/{len(batches)}, size: {batch_config['size_mb']:.2f} MB, records: {len(batch_config['texts'])}")
                batch_file_path = batch_config['file_path']
                
                # Submit and process batch
                batch_id = self.submit_batch(batch_file_path, f"Sentiment analysis batch {idx+1}")
                status = self.wait_for_batch_completion(batch_id)
                
                if status["status"] != "completed":
                    logger.error(f"Batch {batch_id} did not complete successfully: {status}")
                    # Add failed responses
                    for i, text in enumerate(batch_config['texts']):
                        all_responses[f"request-{i + (idx * len(texts) // len(batches))}"] = f"ERROR: Batch error - {status['status']}"
                    continue
                
                # Get results and parse responses
                batch_results = self.get_batch_results(batch_id)
                batch_responses = self.extract_responses_from_batch(batch_results)
                all_responses.update(batch_responses)
                
                # Delete the batch file
                try:
                    os.unlink(batch_file_path)
                except Exception as e:
                    logger.warning(f"Error deleting batch file {batch_file_path}: {str(e)}")
            
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
        """Get sentiment classification using GPT-4.1.
        
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
            # Determine if we need to split the batch based on file size
            batches = self.check_file_size_and_split(texts, translation_prompt_template)
            
            # Process all batches
            all_responses = {}
            
            for idx, batch_config in enumerate(batches):
                logger.info(f"Processing translation batch {idx+1}/{len(batches)}, size: {batch_config['size_mb']:.2f} MB, records: {len(batch_config['texts'])}")
                batch_file_path = batch_config['file_path']
                
                # Submit and process batch
                batch_id = self.submit_batch(batch_file_path, f"{description} {idx+1}")
                status = self.wait_for_batch_completion(batch_id)
                
                if status["status"] != "completed":
                    logger.error(f"Batch {batch_id} did not complete successfully: {status}")
                    # Add failed responses
                    for i, text in enumerate(batch_config['texts']):
                        all_responses[f"request-{i + (idx * len(texts) // len(batches))}"] = f"ERROR: Batch error - {status['status']}"
                    continue
                
                # Get results and parse responses
                batch_results = self.get_batch_results(batch_id)
                batch_responses = self.extract_responses_from_batch(batch_results)
                all_responses.update(batch_responses)
                
                # Delete the batch file
                try:
                    os.unlink(batch_file_path)
                except Exception as e:
                    logger.warning(f"Error deleting batch file {batch_file_path}: {str(e)}")
            
            # Process the responses to extract translations
            translations = []
            for i in range(len(texts)):
                custom_id = f"request-{i}"
                response = all_responses.get(custom_id, f"ERROR: No response for {custom_id}")
                
                # Extract the translation from the response
                if "ERROR:" in response:
                    translations.append(f"TRANSLATION_FAILED: {texts[i]}")
                else:
                    # Check if we need to extract from a formatted response
                    if "Standard American English translation:" in response:
                        parts = response.split("Standard American English translation:")
                        if len(parts) > 1:
                            translations.append(parts[1].strip())
                        else:
                            translations.append(f"TRANSLATION_FAILED: {texts[i]}")
                    elif "African American English translation:" in response:
                        parts = response.split("African American English translation:")
                        if len(parts) > 1:
                            translations.append(parts[1].strip())
                        else:
                            translations.append(f"TRANSLATION_FAILED: {texts[i]}")
                    else:
                        # Just use the whole response
                        translations.append(response.strip())
            
            return translations
            
        except Exception as e:
            logger.error(f"Error in batch translation: {str(e)}")
            return [f"TRANSLATION_FAILED: {str(e)}"] * len(texts)
        finally:
            # Clean up
            if 'batch_file_path' in locals() and os.path.exists(batch_file_path):
                try:
                    os.unlink(batch_file_path)
                except:
                    pass

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
                parts = response.split("Standard American English translation:")
                if len(parts) > 1:
                    return parts[1].strip()
            
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
                parts = response.split("African American English translation:")
                if len(parts) > 1:
                    return parts[1].strip()
            
            # If no specific format found, return the whole response
            return response.strip()
        except Exception as e:
            logger.error(f"Error translating SAE to AAE: {str(e)}")
            return f"TRANSLATION_FAILED: {text}"

def call_gpt41(text: str) -> str:
    """Function to call GPT-4.1 model.
    
    Args:
        text: Input text/prompt
        
    Returns:
        Model response
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    model = GPT41BatchInterface(api_key=api_key)
    return model.call_model(text)