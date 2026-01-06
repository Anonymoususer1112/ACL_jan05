"""
Phi-3-Medium model interface using Hugging Face transformers.
"""

from cProfile import label
import os
import re
from urllib import response

from click import prompt
from gguf import Literal
os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
import logging
import torch
from typing import Dict, List, Any, Optional
import numpy as np
import random

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from .model_interface import TransformersModelInterface

logger = logging.getLogger(__name__)

class Phi3MediumInterface(TransformersModelInterface):
    """Interface for Phi-3-Medium using HuggingFace Transformers."""
    
    def __init__(self, model_path: str = "microsoft/Phi-3-medium-4k-instruct"):
        """Initialize the Phi-3-Medium interface."""
        super().__init__("phi3_medium", model_path)
    
    def _initialize_model(self):
        """Initialize the model and tokenizer."""
        try:
            logger.info(f"Loading Phi-3-Medium from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                trust_remote_code=False,
            ).to(self.device)
            logger.info(f"Successfully loaded Phi-3-Medium")
        except Exception as e:
            logger.error(f"Failed to load Phi-3-Medium: {str(e)}")
            raise
    
    def call_model(self, text: str, max_tokens: int = 512) -> str:
        """Call Phi-3-Medium with a text prompt.
        
        Args:
            text: Input text/prompt
            
        Returns:
            Model response as string
        """
        try:
            # Format as instruction for better results
            messages = [
                {"role": "user", "content": text}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Extract generated text
            generated_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Error calling Phi-3-Medium: {str(e)}")
            return f"ERROR: {str(e)}"
    
    def batch_call_model(self, texts: List[str], batch_size: int = 4) -> List[str]:
        """Efficient batch processing for Phi-3-Medium.
        
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
                # Process each prompt in the batch
                batch_results = []
                for text in batch:
                    # Format as instruction for better results
                    messages = [
                        {"role": "user", "content": text}
                    ]
                    prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                    
                    # Generate response
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=512,
                            temperature=0.3,
                            do_sample=True,
                            top_p=0.9,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                    
                    # Extract generated text
                    generated_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    batch_results.append(generated_text.strip())
                
                results.extend(batch_results)
                #logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                failed_results = ["ERROR: Processing failed"] * len(batch)
                results.extend(failed_results)
            
            # Add delay to avoid overloading GPU
            torch.cuda.empty_cache() if self.device == "cuda" else None
        
        return results
    
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
        """Get sentiment classification using Phi-3-Medium.
        
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


    def get_continuation_preference(
        self,
    )
        
    # def get_preference(self, text_sae: str, text_aae: str, context: Literal["social", "general"] = "social", rng: Optional[random.Random] = None) -> Dict[str, Any]:
    #     """Get preference between two texts using Phi-3-Medium.
        
    #     Args:
    #         text_sae: Standard American English input text
    #         text_aae: African American English input text
    #     Returns:
    #         Preferred text: 'SAE' or 'AAE'
    #     """
    #     r = rng if rng is not None else random
    #     swap = r.random() < 0.5        # swap A and B with 50% probability to avoid position bias
    #     if swap:
            
    #         A_text, B_text = text_aae, text_sae
    #         label_map = {'A': 'AAE', 'B': 'SAE'}
    #     else:
    #         A_text, B_text = text_sae, text_aae
    #         label_map = {'A': 'SAE', 'B': 'AAE'}

    #     instruction = (
    #         "Choose which one reads more appropriate/natural for a more oral context / social media." if context == "social"
    #         else "Choose which one reads more appropriate/natural for general purpose text."
    #     )

    #     prompt = f"""You will see two versions of the same sentence written in different styles of English.\n
    #     {instruction}\n
    #     Return only "A" or "B"\n
    #     Text A: "{A_text}"\n
    #     Text B: "{B_text}"\n
    #     """

    #     try:
    #         response = self.call_model(prompt)
    #         response = response.strip().upper()
    #         if response in ["A", "B", "OPTION A", "OPTION B"]:
    #             if response.startswith("OPTION"):
    #                 response = response[-1]  # Get last character
    #             preferred = label_map[response] # Map back to original labels
    #         else:
    #             logger.warning(f"Unexpected preference response: {response}")
    #             preferred = 'ERROR'
            
    #         return {'preferred': preferred,'A_is': label_map['A'], 'B_is': label_map['B']}
    #     except Exception as e:
    #         logger.error(f"Error getting preference: {str(e)}")
    #         return {'preferred': 'ERROR','A_is': label_map['A'], 'B_is': label_map['B']}




    def get_continuation(self, prefix: str, dialect: Literal["sae", "aae"], max_token: int = 100) -> str:
        """Complete a sentence given a prefix in specified dialect.
        
        Args:
            prefix: Input sentence prefix
            dialect: Target dialect for completion ('sae' or 'aae')
            
        Returns:
            Completed sentence
        """
        prompt = f"""
            Continue the sentence naturally, keeping the same semantic and sentiment. Do not explain; produce only the requested output.\n
            Prefix: "{prefix}"\n
            Continuation:
        """

        try:
            response = self.call_model(prompt, max_tokens=max_token)
            return response.strip()
        except Exception as e:
            logger.error(f"Error completing sentence: {str(e)}")
            return f"ERROR: {str(e)}"
        


        
    def batch_get_sentiment(self, texts: List[str], batch_size: int = 16) -> List[Dict[str, Any]]:
        """Batch sentiment classification using Phi-3-Medium.
        
        Args:
            texts: List of input texts to classify
            batch_size: Size of each batch 

        Returns:
            List of dictionaries with sentiment classifications
        """

        prompts = [
            f"""
            You are a sentiment classifier. Please output ONE label from exactly this set:
            'positive, negative, neutral'. DO NOT output anything else.

            Text: "{text}"
            
            Sentiment:
            """ for text in texts
        ]

        try:
            responses = self.batch_call_model(prompts, batch_size=batch_size)
            results = []
            for response in responses:
                response = response.strip().lower()
                
                # Check if response contains one of the expected sentiments
                if response == 'positive':
                    sentiment = 'positive'
                    score = 1
                elif response == 'negative':
                    sentiment = 'negative'
                    score = -1
                elif response == 'neutral':
                    sentiment = 'neutral'
                    score = 0
                # Fallback to contains if no exact match
                elif 'positive' in response and 'negative' not in response:
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
                
                results.append({
                    'sentiment': sentiment,
                    'score': score,
                    'raw_response': response
                })
            return results
        
        except Exception as e:
            logger.error(f"Error in batch sentiment analysis: {str(e)}")
            return [{'sentiment': 'ERROR', 'score': 0, 'raw_response': str(e)} for _ in texts]


def call_phi3_medium(text: str) -> str:
    """Function to call Phi-3-Medium model.
    
    Args:
        text: Input text/prompt
        
    Returns:
        Model response
    """
    model = Phi3MediumInterface()
    return model.call_model(text)