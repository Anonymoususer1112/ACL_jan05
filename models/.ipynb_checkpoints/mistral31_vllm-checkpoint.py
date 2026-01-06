import os
import re
# os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2'
import logging
import json
import torch
from typing import Dict, List, Any, Optional
from vllm import LLM, SamplingParams
import multiprocessing as mp
from gguf import Literal
import random
from regex import F
import numpy as np
from tqdm import tqdm


# Change relative import to absolute import
try:
    from .model_interface import  ModelInterface
except ImportError:
    import sys
    import os
    # Add parent directory to path
    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    from models.model_interface import  ModelInterface

logger = logging.getLogger(__name__)

class Mistral31VllmInterface(ModelInterface):
    """Interface for Mistral-3.1 using vLLM direct API."""
    
    def __init__(self, model_id: str = "mistralai/Mistral-Small-3.1-24B-Instruct-2503", dtype: str = "float16"):
        """
        Initialize the Mistral-3.1 vLLM interface.

        Args:
            model_name (str): The name of the Mistral-3.1 model to use.
            **kwargs: Additional keyword arguments for future extensions.
        """
        super().__init__("mistral31_vllm")
        self.model_id = model_id
        self.dtype = dtype
        self.llm = None
        self.sampling_params = SamplingParams(temperature=0.5, top_p=1.0, max_tokens=10)
        logger.info(f"Initialized Mistral31VllmInterface with model {self.model_id}")

        # Initialize the LLM (lazy loading - will load on first call)
        self._load_model()

    def _load_model(self):
        if self.llm is None:
            try:
                logger.info(f"Loading model {self.model_id} with dtype {self.dtype}")
                self.llm = LLM(model=self.model_id, dtype=self.dtype, tensor_parallel_size=2, gpu_memory_utilization=0.85,)
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                raise

    def call_model(self, text: str) -> str:
        """Call the Mistral-3.1 model with a given prompt."""
        try:
            # Ensure model is loaded
            self._load_model()
            
            # Format as chat prompt
            prompt = f"<|user|>\n{text}<|assistant|>\n"
            
            # Generate response
            output = self.llm.generate([prompt], self.sampling_params, use_tqdm=False)[0]
            
            # Extract and return response
            return output.outputs[0].text.strip()
                
        except Exception as e:
            logger.error(f"Error calling Mistral3.1 via vLLM: {str(e)}")
            return f"ERROR: {str(e)}"
        
    def batch_call_model(self, texts, batch_size = 8):
        results = []

        # Process in batches to prevent memory issues with very large inputs
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                # Format prompts as chat messages
                prompts = [f"<|user|>\n{text}<|assistant|>\n" for text in batch]
                
                # Generate responses for the batch
                outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
                
                # Extract responses
                batch_results = [output.outputs[0].text.strip() for output in outputs]
                
                results.extend(batch_results)
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                failed_results = ["ERROR: Processing failed"] * len(batch)
                results.extend(failed_results)
                
                # Try to recover GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return results
    
    def get_continuation_preference(
        self, context_sae: str, continuation_sae: str, context_aae: str, continuation_aae: str, context_variant: Literal["SAE", "AAE"] = "SAE", rng: Optional[random.Random] = None, prompt_setting: str = "sae"
    ) -> Dict[str, Any]:
        """
        Compare SAE vs AAE continuations given a fixed context.

        Args:
            context_sae:     Context in Standard American English
            continuation_sae:Continuation in SAE
            context_aae:     Context in African American English
            continuation_aae:Continuation in AAE
            context_variant: Which context to use for this comparison: "SAE" or "AAE"
            setting:         "social" (oral / social media) or "general"

        Returns:
            dict with:
                - preferred: 'SAE' or 'AAE' or 'ERROR'
                - A_is: which dialect A is ('SAE' or 'AAE')
                - B_is: which dialect B is
                - context_variant: which context dialect was used
        """

        r = rng if rng is not None else random

        if context_variant.upper() == "AAE":
            context_text = context_aae
        else:
            context_text = context_sae
            context_variant = "SAE" 

        swap = r.random() < 0.5        # swap A and B with 50% probability to avoid position bias
        if swap:
            A_cont, B_cont = continuation_aae, continuation_sae
            label_map = {"A": "AAE", "B": "SAE"}
        else:
            A_cont, B_cont = continuation_sae, continuation_aae
            label_map = {"A": "SAE", "B": "AAE"}

        # Make sure the prompt is clear and simple to avoid prompt dialect bias.
        if prompt_setting.lower() == "sae":
            prompt = f"""Must pick only one continuation that best follows the context.          
                        Answer only "A" or "B". No explanations.

                        Context: "{context_text}"
                        A: "{A_cont}"
                        B: "{B_cont}"
                        """
        
        # elif prompt_setting.lower() == "aae":
        #     prompt = f"""You gotta choose the one that keep the flow goin’ right. 
        #                 Only "A" or "B".
        #                 Context: "{context_text}"
        #                 A: "{A_cont}"
        #                 B: "{B_cont}"        """
        
        else:
            raise ValueError(f"Unknown prompt_setting: {prompt_setting}")
        
        max_retries = 3
        last_raw_response = ""

        for attempt in range(1, max_retries + 1):
            # On retries, make the instruction even stricter (no extra helper functions)
            if attempt == 1:
                prompt_try = prompt
            else:
                prompt_try = prompt + '\nIMPORTANT: Output exactly ONE character: "A" or "B". Do not output anything else.\n'

            try:
                response = self.call_model(prompt_try)
                raw_response = response
                last_raw_response = raw_response
                response = response.strip().upper()

                # Direct strict match
                if response == "A":
                    choice = "A"
                elif response == "B":
                    choice = "B"
                else:
                    # regex Fallback
                    match = re.search(r"\b([AB])\b", response)
                    if match:
                        choice = match.group(1)
                    else:
                        choice = "ERROR"
                        logger.warning(
                            f"Unexpected continuation preference response (attempt {attempt}/{max_retries}): {raw_response}"
                        )

                # If parsed OK, return immediately
                if choice in label_map:
                    preferred = label_map[choice]
                    return {
                        "preferred": preferred,
                        "A_is": label_map["A"],
                        "B_is": label_map["B"],
                        "raw_response": raw_response,
                    }

                # else: choice == "ERROR" -> retry

            except Exception as e:
                logger.error(f"Error getting continuation preference (attempt {attempt}/{max_retries}): {str(e)}")
                # retry

        # All retries failed
        return {
            "preferred": "ERROR",
            "A_is": label_map["A"],
            "B_is": label_map["B"],
            "raw_response": last_raw_response,
        }
    
    def get_sentiment(self, text: str) -> Dict[str, Any]:
        """Get sentiment classification using Phi-4.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with sentiment classification
        """
        prompt = f"""
        Analyze the sentiment of this text: "{text}"
        
        IMPORTANT: Respond with ONLY ONE WORD - either 'positive', 'negative', or 'neutral'.
        Do not explain your answer. Just output the single word.
        
        Sentiment:
        """
        
        try:
            response = self.call_model(prompt)
            response = response.strip().lower()
            #logger.info(f"Raw response: {response}")
            
            # Improved sentiment detection - checking for exact matches first
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
            
            return {
                'sentiment': sentiment,
                'score': score,
                'raw_response': response
            }
        except Exception as e:
            logger.error(f"Error getting sentiment: {str(e)}")
            return {'sentiment': 'ERROR', 'score': 0, 'raw_response': str(e)}
        
    def batch_get_sentiment(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Get sentiment classification for a batch of texts using Phi-4's batch processing.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of dictionaries with sentiment classifications
        """
        # Create sentiment analysis prompts with stricter instructions
        prompts = [
            f"""
            Analyze the sentiment of this text: "{text}"
            
            IMPORTANT: Respond with ONLY ONE WORD - either 'positive', 'negative', or 'neutral'.
            Do not explain your answer. Just output the single word.
            
            Sentiment:
            """
            for text in texts
        ]
        
        try:
            # Format prompts as chat messages
            chat_prompts = [f"<|user|>\n{prompt}<|assistant|>\n" for prompt in prompts]
            
            # Generate responses for the batch
            outputs = self.llm.generate(chat_prompts, self.sampling_params, use_tqdm=False)
            
            # Process responses
            results = []
            for i, output in enumerate(outputs):
                response = output.outputs[0].text.strip().lower()
                logger.info(f"Raw response: {response}")
                
                # Improved sentiment detection - checking for exact matches first
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
    
def call_mistral31_vllm(text: str) -> str:
    model = Mistral31VllmInterface()
    return model.call_model(text)

def batch_call_mistral31_vllm(texts: List[str], batch_size: int = 8) -> List[str]:
    model = Mistral31VllmInterface()
    return model.batch_call_model(texts, batch_size=batch_size)

