"""
Claude 3 Haiku model interface for Anthropic API.
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
import anthropic

from .model_interface import APIModelInterface

logger = logging.getLogger(__name__)

class ClaudeHaikuInterface(APIModelInterface):
    """Interface for Claude 3 Haiku through Anthropic API."""
    
    def __init__(self, api_key: str = None):
        """Initialize the Claude 3 Haiku interface."""
        super().__init__("claude_haiku", api_key)
        
        # Configure client
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # Set Claude 3 Haiku specific parameters
        self.model_id = "claude-3-haiku-20240307"
        self.rate_limit_per_minute = 50  # Adjust based on API tier
        
        # Configure retries and rate limits
        self.max_retries = 5
        self.retry_delay = 5
        
    @backoff.on_exception(backoff.expo, 
                         (anthropic.RateLimitError, anthropic.APIError), 
                         max_tries=5)
    def call_model(self, text: str) -> str:
        """Call the Claude 3 Haiku model with a text prompt.
        
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
                temperature=0.3,  # Lower temperature for more consistent outputs
                system="You are a helpful assistant.",
                messages=[{"role": "user", "content": text}]
            )
            
            # Extract the response text
            if response.content and len(response.content) > 0:
                text_blocks = [block.text for block in response.content if hasattr(block, 'text')]
                return " ".join(text_blocks)
            else:
                logger.warning("No response from Claude 3 Haiku")
                return "ERROR: No response generated"
                
        except anthropic.RateLimitError as e:
            logger.warning(f"Rate limit exceeded: {str(e)}")
            raise  # Will be caught by backoff decorator
            
        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise  # Will be caught by backoff decorator
            
        except Exception as e:
            logger.error(f"Error calling Claude 3 Haiku: {str(e)}")
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
                        "https://api.anthropic.com/v1/messages",
                        headers={
                            "Content-Type": "application/json",
                            "x-api-key": self.api_key,
                            "anthropic-version": "2023-06-01"
                        },
                        json={
                            "model": self.model_id,
                            "max_tokens": 1000,
                            "temperature": 0.3,
                            "system": "You are a helpful assistant.",
                            "messages": [{"role": "user", "content": text}]
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
                    content_blocks = data.get("content", [])
                    text_blocks = [block.get("text", "") for block in content_blocks if block.get("type") == "text"]
                    return " ".join(text_blocks)
                    
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
        """Get sentiment classification using Claude 3 Haiku.
        
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


def call_claude_haiku(text: str) -> str:
    """Function to call Claude 3 Haiku model.
    
    Args:
        text: Input text/prompt
        
    Returns:
        Model response
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    model = ClaudeHaikuInterface(api_key=api_key)
    return model.call_model(text)