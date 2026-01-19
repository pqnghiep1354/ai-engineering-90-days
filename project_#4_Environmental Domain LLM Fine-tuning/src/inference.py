"""
Inference utilities for Environmental LLM.
"""

from typing import Any, Dict, List, Optional, Union

import torch
from loguru import logger
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    TextStreamer,
    pipeline,
)

from .config import format_instruction, get_instruction_template, get_settings
from .model_utils import load_lora_model, load_model, load_tokenizer


# =============================================================================
# Environmental LLM
# =============================================================================

class EnvironmentalLLM:
    """
    High-level inference interface for Environmental LLM.
    """
    
    def __init__(
        self,
        model_path: str,
        base_model: Optional[str] = None,
        device: str = "auto",
        torch_dtype: str = "float16",
        template_name: str = "alpaca",
        load_in_4bit: bool = False,
    ):
        """
        Initialize Environmental LLM.
        
        Args:
            model_path: Path to fine-tuned model (or LoRA adapter)
            base_model: Base model name (required for LoRA)
            device: Device to use
            torch_dtype: Model precision
            template_name: Instruction template
            load_in_4bit: Load in 4-bit quantization
        """
        self.model_path = model_path
        self.base_model = base_model
        self.device = device
        self.template_name = template_name
        
        logger.info(f"Loading model from: {model_path}")
        
        # Determine if this is a LoRA adapter or full model
        if base_model:
            # Load as LoRA adapter
            self.model, self.tokenizer = load_lora_model(
                base_model_name=base_model,
                lora_path=model_path,
                device_map=device,
                torch_dtype=torch_dtype,
            )
            # Merge for faster inference
            self.model = self.model.merge_and_unload()
        else:
            # Load as full model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                torch_dtype=getattr(torch, torch_dtype),
                trust_remote_code=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
        
        # Set to eval mode
        self.model.eval()
        
        # Default generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        logger.info("Model loaded successfully")
    
    def generate(
        self,
        instruction: str,
        input_text: str = "",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        stream: bool = False,
    ) -> str:
        """
        Generate response for an instruction.
        
        Args:
            instruction: The instruction/question
            input_text: Optional input context
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample
            stream: Whether to stream output
            
        Returns:
            Generated response
        """
        # Format prompt
        prompt = format_instruction(
            instruction=instruction,
            input_text=input_text,
            template_name=self.template_name,
        )
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device)
        
        # Generation config
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Generate
        with torch.no_grad():
            if stream:
                streamer = TextStreamer(self.tokenizer, skip_prompt=True)
                outputs = self.model.generate(
                    **inputs,
                    generation_config=gen_config,
                    streamer=streamer,
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    generation_config=gen_config,
                )
        
        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        
        return response.strip()
    
    def batch_generate(
        self,
        instructions: List[str],
        input_texts: Optional[List[str]] = None,
        **kwargs,
    ) -> List[str]:
        """
        Generate responses for multiple instructions.
        
        Args:
            instructions: List of instructions
            input_texts: Optional list of input contexts
            **kwargs: Generation parameters
            
        Returns:
            List of responses
        """
        if input_texts is None:
            input_texts = [""] * len(instructions)
        
        responses = []
        for instruction, input_text in zip(instructions, input_texts):
            response = self.generate(instruction, input_text, **kwargs)
            responses.append(response)
        
        return responses
    
    def chat(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: str = "You are an expert in environmental science and climate change.",
        **kwargs,
    ) -> str:
        """
        Chat interface with conversation history.
        
        Args:
            message: User message
            history: Conversation history
            system_prompt: System prompt
            **kwargs: Generation parameters
            
        Returns:
            Assistant response
        """
        # Build conversation context
        context_parts = [f"System: {system_prompt}"]
        
        if history:
            for turn in history[-5:]:  # Keep last 5 turns
                context_parts.append(f"User: {turn.get('user', '')}")
                context_parts.append(f"Assistant: {turn.get('assistant', '')}")
        
        context = "\n".join(context_parts)
        
        # Generate response
        response = self.generate(
            instruction=message,
            input_text=context,
            **kwargs,
        )
        
        return response


# =============================================================================
# Pipeline Wrapper
# =============================================================================

def create_pipeline(
    model_path: str,
    base_model: Optional[str] = None,
    task: str = "text-generation",
    device: int = 0,
    **kwargs,
):
    """
    Create HuggingFace pipeline for inference.
    
    Args:
        model_path: Path to model
        base_model: Base model for LoRA
        task: Pipeline task
        device: Device index
        **kwargs: Additional pipeline arguments
        
    Returns:
        Pipeline instance
    """
    if base_model:
        model, tokenizer = load_lora_model(
            base_model_name=base_model,
            lora_path=model_path,
        )
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
    
    return pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        device=device,
        **kwargs,
    )


# =============================================================================
# Batch Inference
# =============================================================================

class BatchInference:
    """
    Efficient batch inference for large-scale processing.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 8,
        max_length: int = 2048,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
    
    def process(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        **gen_kwargs,
    ) -> List[str]:
        """
        Process multiple prompts efficiently.
        
        Args:
            prompts: List of prompts
            max_new_tokens: Max tokens to generate
            **gen_kwargs: Generation parameters
            
        Returns:
            List of generated responses
        """
        responses = []
        
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i + self.batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **gen_kwargs,
                )
            
            # Decode
            for j, output in enumerate(outputs):
                response = self.tokenizer.decode(
                    output[inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                responses.append(response.strip())
        
        return responses


# =============================================================================
# Evaluation Helpers
# =============================================================================

def evaluate_response(
    response: str,
    reference: str,
    metrics: List[str] = ["exact_match", "contains_key"],
) -> Dict[str, float]:
    """
    Simple evaluation of generated response.
    
    Args:
        response: Generated response
        reference: Reference answer
        metrics: Metrics to compute
        
    Returns:
        Dictionary of metric scores
    """
    results = {}
    
    response_lower = response.lower().strip()
    reference_lower = reference.lower().strip()
    
    if "exact_match" in metrics:
        results["exact_match"] = float(response_lower == reference_lower)
    
    if "contains_key" in metrics:
        # Check if key phrases from reference are in response
        key_phrases = reference_lower.split()[:5]  # First 5 words
        matches = sum(1 for kp in key_phrases if kp in response_lower)
        results["contains_key"] = matches / len(key_phrases) if key_phrases else 0.0
    
    if "length_ratio" in metrics:
        results["length_ratio"] = len(response) / max(len(reference), 1)
    
    return results
