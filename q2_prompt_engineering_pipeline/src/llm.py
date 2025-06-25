"""Module for interacting with language models."""
import os
from typing import List, Dict, Any, Optional, Union
import json
import time
import google.generativeai as genai
from pathlib import Path
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryCallState
)

from .config import get_config
from .models import Task, ReasoningPath
from .utils import calculate_md5, get_logger

class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class LLMResponseError(LLMError):
    """Exception raised when there's an error in the LLM response."""
    pass


class LLMClient:
    """Client for interacting with Google's Gemini 2.0 Flash model."""
    
    def __init__(
        self, 
        model: Optional[str] = None, 
        api_key: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the Gemini client.
        
        Args:
            model: Model name (defaults to config.DEFAULT_MODEL)
            api_key: Google API key (defaults to GOOGLE_API_KEY environment variable)
            generation_config: Configuration for text generation
        """
        self.config = get_config()
        self.model_name = model or self.config.DEFAULT_MODEL
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "Google API key not provided. Set GOOGLE_API_KEY environment variable "
                "or pass api_key to LLMClient constructor."
            )
            
        # Configure the Gemini client
        genai.configure(api_key=self.api_key)
        
        # Set up the model with generation config
        self.generation_config = generation_config or {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config
        )
        
        self.logger = get_logger("GeminiClient")
        self.logger.info(f"Initialized Gemini client with model: {self.model_name}")
    
    def _prepare_content(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        examples: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """Prepare content for the Gemini API.
        
        Args:
            prompt: The user's prompt
            system_message: Optional system message
            examples: Optional list of example messages
            
        Returns:
            List of message dictionaries
        """
        messages = []
        
        # Add system message if provided
        if system_message:
            messages.append({"role": "user", "parts": [system_message]})
            messages.append({"role": "model", "parts": ["Understood."]})
        
        # Add examples if provided
        if examples:
            for example in examples:
                if "user" in example and "assistant" in example:
                    messages.extend([
                        {"role": "user", "parts": [example["user"]]},
                        {"role": "model", "parts": [example["assistant"]]}
                    ])
        
        # Add the actual user prompt
        messages.append({"role": "user", "parts": [prompt]})
        
        return messages
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((
            ConnectionError,
            TimeoutError,
            LLMError
        )),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying LLM call (attempt {retry_state.attempt_number}): {retry_state.outcome.exception()}"
        ) if retry_state.outcome else None
    )
    def generate_completion(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        examples: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate a completion using the Gemini API.
        
        Args:
            prompt: The prompt to complete
            system_message: Optional system message
            examples: Optional list of example messages
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            The generated text
        """
        try:
            # Update generation config if parameters are provided
            config_updates = {}
            if temperature is not None:
                config_updates["temperature"] = temperature
            if max_tokens is not None:
                config_updates["max_output_tokens"] = max_tokens
                
            if config_updates:
                self.model.generation_config = genai.types.GenerationConfig(
                    **{**self.generation_config, **config_updates}
                )
            
            # Prepare messages
            messages = self._prepare_content(prompt, system_message, examples)
            
            # Generate content
            response = self.model.generate_content(messages)
            print(response,"THIS IS THE RESPONSE")
            
            # Extract and return the generated text
            if response.candidates and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text
            else:
                raise LLMResponseError("No content in response")
                
        except Exception as e:
            self.logger.error(f"Error generating completion: {e}")
            raise LLMError(f"Failed to generate completion: {e}") from e
    
    def generate_reasoning_paths(
        self,
        task: Task,
        num_paths: int = 3,
        temperature: float = 0.8
    ) -> List[ReasoningPath]:
        """Generate multiple reasoning paths for a task.
        
        Args:
            task: The task to generate reasoning paths for
            num_paths: Number of reasoning paths to generate
            temperature: Sampling temperature (higher = more diverse)
            
        Returns:
            List of ReasoningPath objects
        """
        self.logger.info(f"Generating {num_paths} reasoning paths for task: {task.id}")
        
        # Prepare the prompt for generating reasoning paths
        system_message = (
            "You are an expert problem solver. Generate different approaches to solve "
            "the given problem, each with a clear step-by-step reasoning process. "
            "Always respond with valid JSON in the specified format."
        )
        
        prompt = (
            f"Task: {task.description}\n\n"
            f"Generate {num_paths} different approaches to solve this problem. "
            "For each approach, provide a clear, step-by-step reasoning process "
            "and a final answer. Format your response as a JSON array of objects "
            "with 'reasoning' and 'answer' fields. Example output format:\n"
            "[{\"reasoning\": \"Step 1... Step 2...\", \"answer\": \"final answer\"}]"
        )
        
        try:
            # Generate paths using the Gemini model
            response = self.generate_completion(
                prompt=prompt,
                system_message=system_message,
                temperature=temperature
            )
            
            # Clean and parse the response
            try:
                # Try to extract JSON from markdown code blocks
                import re
                json_match = re.search(r'```(?:json)?\n(.*?)\n```', response, re.DOTALL)
                if json_match:
                    response = json_match.group(1)
                
                # Parse the JSON
                paths_data = json.loads(response)
                
                # Handle different response formats
                if isinstance(paths_data, dict) and "paths" in paths_data:
                    paths_data = paths_data["paths"]
                elif not isinstance(paths_data, list):
                    paths_data = [paths_data]
                
                # Convert to ReasoningPath objects
                reasoning_paths = []
                for i, path_data in enumerate(paths_data[:num_paths], 1):
                    if not isinstance(path_data, dict):
                        continue
                        
                    reasoning = path_data.get("reasoning", "")
                    answer = path_data.get("answer", "")
                    
                    if not reasoning or not answer:
                        continue
                    
                    # Split reasoning into steps
                    steps = [
                        step.strip() for step in reasoning.split('\n') 
                        if step.strip()
                    ]
                    
                    # Calculate confidence (simple heuristic based on reasoning length)
                    confidence = min(0.95, 0.5 + (len(steps) / 20))
                    
                    reasoning_paths.append(ReasoningPath(
                        id=f"{task.id}_path_{i}",
                        task_id=task.id,
                        path=steps,
                        final_answer=answer,
                        confidence=round(confidence, 2),
                        metadata={
                            "model": self.model_name,
                            "generator": "gemini",
                            "timestamp": time.time()
                        }
                    ))
                
                self.logger.info(f"Generated {len(reasoning_paths)} reasoning paths")
                return reasoning_paths
                
            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                self.logger.error(f"Error parsing reasoning paths: {e}")
                self.logger.debug(f"Response content: {response}")
                raise LLMResponseError(f"Failed to parse reasoning paths: {e}") from e
                
        except Exception as e:
            self.logger.error(f"Error generating reasoning paths: {e}")
            raise LLMError(f"Failed to generate reasoning paths: {e}") from e
    
    def evaluate_solution(
        self,
        task: Task,
        solution: str,
        criteria: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluate a solution to a task.
        
        Args:
            task: The task being solved
            solution: The proposed solution
            criteria: List of evaluation criteria
            
        Returns:
            Dictionary with evaluation results
        """
        criteria = criteria or ["correctness", "clarity", "completeness"]
        
        system_message = (
            "You are an expert evaluator. Your task is to evaluate solutions "
            "based on the given criteria and provide detailed feedback."
        )
        
        prompt = (
            f"Task: {task.description}\n\n"
            f"Expected Answer: {task.expected_answer}\n\n"
            f"Proposed Solution: {solution}\n\n"
            f"Evaluate this solution based on the following criteria:\n"
            + "\n".join(f"- {criterion}" for criterion in criteria)
            + "\n\nProvide your evaluation as a JSON object with the following fields:\n"
            "- 'score' (0-10): Overall score\n"
            "- 'is_correct' (boolean): Whether the solution is correct\n"
            "- 'feedback' (string): Detailed feedback\n"
            "- 'scores' (object): Individual criterion scores"
        )
        
        try:
            response = self.generate_completion(
                prompt=prompt,
                system_message=system_message,
                temperature=0.2,  # Lower temperature for more consistent evaluations
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            try:
                if '```json' in response:
                    response = response.split('```json')[1].split('```')[0].strip()
                
                evaluation = json.loads(response)
                return {
                    "score": evaluation.get("score", 0),
                    "is_correct": evaluation.get("is_correct", False),
                    "feedback": evaluation.get("feedback", ""),
                    "scores": evaluation.get("scores", {})
                }
                
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.error(f"Error parsing evaluation: {e}")
                self.logger.debug(f"Response content: {response}")
                raise LLMResponseError(f"Failed to parse evaluation: {e}") from e
                
        except Exception as e:
            self.logger.error(f"Error evaluating solution: {e}")
            raise LLMError(f"Failed to evaluate solution: {e}") from e
