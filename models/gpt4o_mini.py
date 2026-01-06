"""
GPT-4o-mini model interface for OpenAI API.
"""

import os
import json
import logging
import asyncio
import aiohttp
import time
import backoff
from typing import Dict, List, Any, Optional

import httpx
import openai

from .model_interface import APIModelInterface

logger = logging.getLogger(__name__)

class GPT4oMiniInterface(APIModelInterface):
    """Interface for GPT-4o-mini through OpenAI API."""
    
    def __init__(self, api_key: str = None):
        """Initialize the GPT-4o-mini interface."""
        super().__init__("gpt4o_mini", api_key)
        
        # Configure client
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Set GPT-4o-mini specific parameters
        self.model_id = "gpt-4o-mini"
        self.rate_limit_per_minute = 60  # Adjust based on API tier
        
        # Configure retries and rate limits
        self.max_retries = 5
        self.retry_delay = 5
        
    @backoff.on_exception(backoff.expo, 
                         (openai.RateLimitError, openai.APIError), 
                         max_tries=5)
    def call_model(self, text: str) -> str:
        """Call the GPT-4o-mini model with a text prompt.
        
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
                logger.warning("No response from GPT-4o-mini")
                return "ERROR: No response generated"
                
        except openai.RateLimitError as e:
            logger.warning(f"Rate limit exceeded: {str(e)}")
            raise  # Will be caught by backoff decorator
            
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise  # Will be caught by backoff decorator
            
        except Exception as e:
            logger.error(f"Error calling GPT-4o-mini: {str(e)}")
            return f"ERROR: {str(e)}"
    
    async def async_call_model(self, text: str) -> str:
        """Async version of call_model for improved performance.
        
        Args:
            text: Input text/prompt
            
        Returns:
            Model response as string
        """
        # No built-in async in official client, so we'll use httpx
        async with httpx.AsyncClient() as client:
            for attempt in range(self.max_retries):
                try:
                    response = await client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.api_key}"
                        },
                        json={
                            "model": self.model_id,
                            "messages": [
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": text}
                            ],
                            "max_tokens": 1000,
                            "temperature": 0.3
                        },
                        timeout=60.0
                    )
                    
                    if response.status_code == 429:
                        retry_after = int(response.headers.get("Retry-After", self.retry_delay))
                        logger.warning(f"Rate limit exceeded, retrying in {retry_after} seconds")
                        await asyncio.sleep(retry_after)
                        continue
                        
                    if response.status_code != 200:
                        logger.error(f"API error: {response.status_code} - {response.text}")
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(self.retry_delay)
                            continue
                        return f"ERROR: API returned {response.status_code}"
                        
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
                    
                except Exception as e:
                    logger.error(f"Error in async call: {str(e)}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay)
                    else:
                        return f"ERROR: {str(e)}"
                        
        return "ERROR: Max retries exceeded"

    def translate_aae_to_sae(self, text: str) -> str:
        """Translate text from AAE to SAE.
        
        Args:
            text: Input text in AAE
            
        Returns:
            Translated text in SAE
        """
        prompt = f"""
        Translate the following text from African American English (AAE) to Standard American English (SAE).
        Preserve the meaning, tone, and intent of the original text.
        Only change dialectical features while maintaining the original message.
        
        Original text (AAE): "{text}"
        
        Standard American English translation:
        """
        
        return self.call_model(prompt)
    
    def translate_sae_to_aae(self, text: str) -> str:
        """Translate text from SAE to AAE.
        
        Args:
            text: Input text in SAE
            
        Returns:
            Translated text in AAE
        """
        prompt = f"""
        Translate the following text from Standard American English (SAE) to African American English (AAE).
        Preserve the meaning, tone, and intent of the original text.
        Only change dialectical features while maintaining the original message.
        
        Original text (SAE): "{text}"
        
        African American English translation:
        """
        
        return self.call_model(prompt)
    
    def get_sentiment(self, text: str) -> Dict[str, Any]:
        """Get sentiment classification using GPT-4o-mini.
        
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


def call_gpt4o_mini(text: str) -> str:
    """Function to call GPT-4o-mini model.
    
    Args:
        text: Input text/prompt
        
    Returns:
        Model response
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    model = GPT4oMiniInterface(api_key=api_key)
    return model.call_model(text)