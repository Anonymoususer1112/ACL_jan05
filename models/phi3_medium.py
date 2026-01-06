"""
Phi-3-Medium model interface using Hugging Face transformers.
"""
from __future__ import annotations


from cProfile import label
import os
import re
from urllib import response

from cv2 import log
from gguf import Literal
from regex import F
from sympy import im
# os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
import logging
import torch
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import random
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from .model_interface import TransformersModelInterface

logger = logging.getLogger(__name__)

PairType = Union[Tuple[str, str], Dict[str, str]]  # (sae, aae) or {"sae":..., "aae":...}

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
            self.device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
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
            prompt = f"""Must pick one continuation that best follows the context.          
                        Answer only "A" or "B".

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
        

    def _get_log_prob_for_text(self, text: str) -> float:
        """
        Calculates the log probability (log P(Text)) of a given text using the model.
        This uses the model's forward pass to get logits and torch.log_softmax.
        
        Args:
            text: The text/sentence for which to calculate the probability.
            
        Returns:
            The total log probability of the text as a float.
        """
        # 1. Tokenize the text
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=True).to(self.model.device)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        if input_ids.shape[1] < 2:
             logger.warning(f"Input text too short for log probability: {text}")
             return -float('inf') # Return a very low value for empty/short texts

        # 2. Get the model's logits
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=input_ids # Setting labels=input_ids calculates the loss for the tokens
            )
            # Logits are of shape (batch_size, sequence_length, vocab_size)
            logits = outputs.logits

        # Calculate log probabilities per token
        # We shift the tokens because the probability of token[i] is predicted by context up to token[i-1]
        # Logits: P(token_i | token_0...token_{i-1})
        # Labels: token_1 to token_N
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        
        # Log probability of the *next* token
        target_log_probs = torch.gather(
            log_probs[:, :-1, :], 
            2, 
            input_ids[:, 1:].unsqueeze(2)
        ).squeeze(2)
        
        # Apply attention mask to ignore padding tokens (if any, though inputs usually aren't padded here)
        mask = attention_mask[:, 1:]
        total_log_prob = (target_log_probs * mask).sum().item()
    
        
        return total_log_prob

    def calculate_log_difference_for_preference(
        self, 
        context_sae: str, 
        continuation_sae: str, 
        context_aae: str, 
        continuation_aae: str,
        context_variant: str = "SAE" 
    ) -> float:
        """
        Calculates Log-Difference based on the context variant used in the preference task.
        """
        
        # 1. Determine the actual context used (C)
        if context_variant.upper() == "AAE":
            context_used = str(context_aae)
        else:
            context_used = str(context_sae)
        
        # 2. Construct the full sentences (S_SAE and S_AAE) and ensure continuations are strings
        sae_text_full = context_used + " " + str(continuation_sae)
        aae_text_full = context_used + " " + str(continuation_aae)
        
        # 3. Use the existing core function to calculate the score
        # Note: We assume the existing self._get_log_prob_for_text uses the raw text, 
        # so we should use the implementation from the previous answer.
        
        log_prob_sae = self._get_log_prob_for_text(sae_text_full)
        log_prob_aae = self._get_log_prob_for_text(aae_text_full)
        
        log_difference = log_prob_sae - log_prob_aae
        
        return log_prob_aae, log_prob_sae, log_difference

        
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
    

    

        
    def get_sentence_activation(self, text: str, hs_idx: int = -1, max_length: int = 128) -> torch.Tensor:
        """
        Returns a single sentence embedding from hidden_states[hs_idx] using masked mean pooling.
        """
        self.model.eval()
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=False,  # single sentence -> no need to pad
        ).to(self.model.device)

        def masked_mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            """
            hidden_states: [B, T, D]
            attention_mask: [B, T] with 1 for real tokens, 0 for padding
            returns: [B, D]
            """
            mask = attention_mask.unsqueeze(-1).type_as(hidden_states)  # [B, T, 1]
            summed = (hidden_states * mask).sum(dim=1)                  # [B, D]
            denom = mask.sum(dim=1).clamp(min=1.0)                      # [B, 1]
            return summed / denom

        with torch.inference_mode():
            out = self.model(**enc, output_hidden_states=True)
            hs = out.hidden_states[hs_idx]  # [1, T, D]
            sent = masked_mean_pool(hs, enc["attention_mask"])[0]   # [D]
            return sent.detach().cpu()

    def make_all_hidden_state_indices(self) -> List[int]:
        """
        Return all valid hidden_state indices.
        Usually: 0 ... num_layers
        (0 = embedding output, num_layers = final layer output)
        """
        num_layers = self.get_num_layers()
        return list(range(1, num_layers + 1)) 

    def get_num_layers(self) -> int:
        """Get the number of layers in the model.
        
        Returns:
            Number of transformer layers
        """
        if hasattr(self.model.config, 'num_hidden_layers'):
            return self.model.config.num_hidden_layers
        
        # Some models may have different naming mechanisms
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return len(self.model.model.layers)
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "layers"):
            return len(self.model.transformer.layers)
        raise ValueError("Cannot infer number of layers for this model.")
    
    def compute_aae_dialect_vector_for_layer(
        self,
        pairs: List[PairType],
        hs_idx: int,
        max_pairs: int | None = None,
        max_length: int = 128,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        if max_pairs is not None:
            pairs = pairs[:max_pairs]

        sae_acts, aae_acts = [], []
        def _unpack_pair(p: PairType) -> Tuple[str, str]:
            if isinstance(p, dict):
                return p["sae"], p["aae"]
            return p[0], p[1]

        for p in tqdm(pairs, desc=f"Computing activations for hidden_states[{hs_idx}]"):
            sae_text, aae_text = _unpack_pair(p)
            a_vec = self.get_sentence_activation(aae_text, hs_idx=hs_idx, max_length=max_length)
            s_vec = self.get_sentence_activation(sae_text, hs_idx=hs_idx, max_length=max_length)
            aae_acts.append(a_vec)
            sae_acts.append(s_vec)

        sae_acts = torch.stack(sae_acts)  # [N, D]
        aae_acts = torch.stack(aae_acts)  # [N, D]

        v_aae = (aae_acts - sae_acts).mean(dim=0)                   # [D]
        v_aae_unit = v_aae / (v_aae.norm() + 1e-8)

        proj_sae = sae_acts @ v_aae_unit                            # [N]
        proj_aae = aae_acts @ v_aae_unit                            # [N]

        stats = {
            "proj_mean_SAE": proj_sae.mean().item(),
            "proj_std_SAE": proj_sae.std(unbiased=False).item(),
            "proj_mean_AAE": proj_aae.mean().item(),
            "proj_std_AAE": proj_aae.std(unbiased=False).item(),
            "mean_gap_AAE_minus_SAE": (proj_aae - proj_sae).mean().item(),
        }

        return v_aae_unit, stats 
    

    def layer_sweep_dialect_vectors_full(
        self,
        pairs: List[Tuple[str, str]],
        max_pairs: int = 200,
        max_length: int = 128,
    ) -> Tuple[Dict[int, Dict[str, Any]], int]:

        num_layers = self.get_num_layers()
        logger.info(f"Model has {num_layers} transformer layers.")

        candidate_layers = list(range(1, num_layers + 1))
        logger.info(f"Sweeping all hidden_state indices: {candidate_layers}")

        results = {}

        for hs_idx in tqdm(candidate_layers, desc="Full layer sweep (dialect vectors)"):
            v_aae, stats = self.compute_aae_dialect_vector_for_layer(
                pairs=pairs,
                hs_idx=hs_idx,
                max_pairs=max_pairs,
                max_length=max_length
            )
            results[hs_idx] = {
                "v_aae_unit": v_aae,
                "stats": stats
            }

        best_layer = max(
            results,
            key=lambda i: results[i]["stats"]["mean_gap_AAE_minus_SAE"]
        )

        logger.info(
            f"Best hidden_state index: {best_layer} "
            f"(gap={results[best_layer]['stats']['mean_gap_AAE_minus_SAE']:.4f})"
        )

        return results, best_layer

def call_phi3_medium(text: str) -> str:
    """Function to call Phi-3-Medium model.
    
    Args:
        text: Input text/prompt
        
    Returns:
        Model response
    """
    model = Phi3MediumInterface()
    return model.call_model(text)