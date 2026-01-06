import os
import re
import logging
import json
import torch
import random
from typing import Dict, List, Any, Optional
from vllm import LLM, SamplingParams
from gguf import Literal

# Change relative import to absolute import logic
try:
    from .model_interface import  ModelInterface
except ImportError:
    import sys
    # Add parent directory to path
    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    from models.model_interface import  ModelInterface

logger = logging.getLogger(__name__)

class DeepseekR1VllmInterface(ModelInterface):
    """Interface for Deepseek-R1 using vLLM direct API, optimized for Reasoning/CoT models."""
    
    def __init__(self, model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", dtype: str = "float16"):
        """
        Initialize the Deepseek-R1 vLLM interface.

        Args:
            model_id (str): The name/path of the Deepseek-R1 model to use.
            dtype (str): The data type for model weights.
        """
        super().__init__("deepseekR1_vllm")
        self.model_id = model_id
        self.dtype = dtype
        self.llm = None
        
        # DeepSeek-R1 configuration:
        # Temperature 0.6 is often recommended for reasoning models to allow diverse thought paths.
        # max_tokens must be large (e.g., 4096) to accommodate the verbose <think> process.
        self.sampling_params = SamplingParams(
            temperature=0.5,
            top_p=1.0,
            max_tokens=150, # Note: 100 might be too short for R1 if it thinks a lot. Consider increasing if you get truncated outputs.
        )
        logger.info(f"Initialized DeepseekR1VllmInterface with model {self.model_id}")

        # Initialize the LLM (lazy loading - will load on first call)
        self._load_model()

    def _format_chat(self, user_text: str) -> str:
        messages = [{"role": "user", "content": user_text}]
        # Ensure tokenizer is available
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            self.tokenizer = self.llm.get_tokenizer()
            
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt

    def _load_model(self):
        if self.llm is None:
            try:
                # Get the actual number of available GPUs
                available_gpus = torch.cuda.device_count()
                
                logger.info(f"Loading model {self.model_id} with dtype {self.dtype}")
                logger.info(f"Detected {available_gpus} GPUs. Setting tensor_parallel_size={available_gpus}")
                
                self.llm = LLM(
                    model=self.model_id, 
                    dtype=self.dtype, 
                    tensor_parallel_size=available_gpus, 
                    gpu_memory_utilization=0.90,
                    trust_remote_code=True
                )
                self.tokenizer = self.llm.get_tokenizer()
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                raise

    def call_model(self, text: str) -> str:
        self._load_model()
        prompt = self._format_chat(text)
        outputs = self.llm.generate([prompt], self.sampling_params, use_tqdm=False)
        generated_text = outputs[0].outputs[0].text.strip()
        return generated_text

    def batch_call_model(self, texts, batch_size=8):
        self._load_model()
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # Format all prompts in batch
            prompts = [self._format_chat(t) for t in batch]
            try:
                outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
                results.extend([out.outputs[0].text.strip() for out in outputs])
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                results.extend(["ERROR"] * len(batch))
        return results

    # --------- Reasoning Extraction Logic ---------

    def _extract_answer(self, full_text: str, candidates: List[str]) -> str:
        """
        Extract the final answer from text that includes <think> tags.
        """
        # 1. Strip thought process to avoid matching A/B inside the reasoning steps
        clean_text = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL).strip()
        
        # 2. Try matching explicit formats first
        # Pattern covers: "Answer: A", "Option A", "Choice: A"
        for c in candidates:
            pattern = fr"(?:Answer|Option|Choice)\s*:?\s*({c})\b"
            if re.search(pattern, clean_text, re.IGNORECASE):
                return c

        # 3. Fallback: Find the last valid candidate mentioned
        # e.g., "Therefore, I choose B." -> returns B
        found_candidates = []
        for c in candidates:
            # Find all occurrences as whole words
            matches = list(re.finditer(fr"\b{c}\b", clean_text, re.IGNORECASE))
            if matches:
                last_pos = matches[-1].start()
                found_candidates.append((last_pos, c))
        
        if found_candidates:
            # Sort by position, return the one that appears last
            found_candidates.sort(key=lambda x: x[0])
            return found_candidates[-1][1]

        return "ERROR"

    # --------- Task Methods ---------

    def get_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using reasoning model with retry logic.
        """

        base_prompt = f"""
        Analyze the sentiment of this text: "{text}"
        
        IMPORTANT: Respond with ONLY ONE WORD - either 'positive', 'negative', or 'neutral'.
        Do not explain your answer. Output your answer first if you provide reasoning.
        
        Answer:
        """

        mapping = {
            "Positive": ("positive", 1),
            "Negative": ("negative", -1),
            "Neutral": ("neutral", 0)
        }

        max_retries = 3
        last_raw_response = ""

        for attempt in range(1, max_retries + 1):
            prompt = base_prompt
            if attempt > 1:
                prompt += (
                    "\n\nReminder: Your final line MUST be exactly one of:\n"
                    "Answer: Positive\nAnswer: Negative\nAnswer: Neutral"
                )

            try:
                raw_response = self.call_model(prompt)
                last_raw_response = raw_response

                answer = self._extract_answer(raw_response, ["Positive", "Negative", "Neutral"])
                
                if answer in mapping:
                    s, sc = mapping[answer]
                    return {
                        "sentiment": s,
                        "score": sc,
                        "raw_response": raw_response,
                    }

                logger.warning(
                    f"Attempt {attempt}/{max_retries}: Failed to parse sentiment. "
                    f"Response end: {raw_response[-100:]}"
                )

            except Exception as e:
                logger.error(
                    f"Attempt {attempt}/{max_retries}: Error in get_sentiment: {str(e)}"
                )

        # All retries failed
        return {
            "sentiment": "ERROR",
            "score": 0,
            "raw_response": last_raw_response[:200]
        }


    def get_continuation_preference(
        self,
        context_sae: str,
        continuation_sae: str,
        context_aae: str,
        continuation_aae: str,
        context_variant: Literal["SAE", "AAE"] = "SAE",
        rng: Optional[random.Random] = None,
        prompt_setting: str = "sae",
    ) -> Dict[str, Any]:
        
        r = rng if rng is not None else random

        # Set context text
        if context_variant.upper() == "AAE":
            context_text = context_aae
            context_variant = "AAE"
        else:
            context_text = context_sae
            context_variant = "SAE"

        # Random swap logic to avoid positional bias
        swap = r.random() < 0.5
        if swap:
            A_cont, B_cont = continuation_aae, continuation_sae
            label_map = {"A": "AAE", "B": "SAE"}
        else:
            A_cont, B_cont = continuation_sae, continuation_aae
            label_map = {"A": "SAE", "B": "AAE"}

        # Base Prompt
        base_prompt = (
            f"""You must choose exactly one continuation that best follows the context.
Reply with only one character: A or B.
            Context: {context_text}\n\n
            Option A: {A_cont}\n
            Option B: {B_cont}\n\n
            IMPORTANT: End your response with exactly: 'Answer: A' or 'Answer: B'.
            """
        )

        max_retries = 3
        last_full_response = ""

        for attempt in range(1, max_retries + 1):
            
            # On retries, add a stricter reminder
            current_prompt = base_prompt
            if attempt > 1:
                current_prompt += "\n\nVerify your format. Did you end with 'Answer: A' or 'Answer: B'?"

            try:
                # 1. Generate full response (including <think>)
                full_response = self.call_model(current_prompt)
                last_full_response = full_response
                
                # 2. Extract answer using regex helper
                choice = self._extract_answer(full_response, ["A", "B"])
                
                # 3. Valid extraction? Return immediately
                if choice in label_map:
                    return {
                        "preferred": label_map[choice],
                        "A_is": label_map["A"],
                        "B_is": label_map["B"],
                        "raw_response": choice,
                        "context_variant": context_variant,
                        "full_response_snippet": full_response[:200] + "..." 
                    }
                else:
                    # Logic failed to extract -> Retry
                    logger.warning(f"Attempt {attempt}/{max_retries}: Failed to extract A/B. Response end: {full_response[-100:]}")

            except Exception as e:
                logger.error(f"Attempt {attempt}/{max_retries}: Error in get_continuation_preference: {str(e)}")
                # Exception -> Retry

        # If all retries fail, return ERROR
        return {
            "preferred": "ERROR",
            "A_is": label_map["A"],
            "B_is": label_map["B"],
            "raw_response": "MAX_RETRIES_EXCEEDED",
            "context_variant": context_variant,
            "full_response": last_full_response
        }

# Helper functions exposed at module level
def call_deepseekr1_vllm(text: str) -> str:
    model_interface = DeepseekR1VllmInterface()
    return model_interface.call_model(text)

def batch_call_deepseekr1_vllm(texts: List[str], batch_size: int = 8) -> List[str]:
    model_interface = DeepseekR1VllmInterface()
    return model_interface.batch_call_model(texts, batch_size=batch_size)