import os
import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

from .models import Task, TaskType, ReasoningPath, PromptVersion
from .reasoning import ReasoningEngine
from .prompt_optimizer import PromptOptimizer
from .evaluation import Evaluator, EvaluationResult

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
)
logger.add(
    "logs/pipeline.log",
    rotation="10 MB",
    retention="1 month",
    level="DEBUG"
)

class Pipeline:
    """Main pipeline for the prompt engineering system."""
    
    def __init__(self, 
                 num_reasoning_paths: int = 3,
                 reasoning_engine: Optional[ReasoningEngine] = None,
                 prompt_optimizer: Optional[PromptOptimizer] = None,
                 evaluator: Optional[Evaluator] = None):
        """Initialize the pipeline.
        
        Args:
            num_reasoning_paths: Number of parallel reasoning paths to generate
            reasoning_engine: Optional custom reasoning engine instance
            prompt_optimizer: Optional custom prompt optimizer instance
            evaluator: Optional custom evaluator instance
        """
        self.reasoning_engine = reasoning_engine or ReasoningEngine(num_paths=num_reasoning_paths)
        self.prompt_optimizer = prompt_optimizer or PromptOptimizer()
        self.evaluator = evaluator or Evaluator()
        self.current_task: Optional[Task] = None
        self.current_paths: List[ReasoningPath] = []
        self.best_path: Optional[ReasoningPath] = None
        self.current_prompt: Optional[PromptVersion] = None
    
    def load_task(self, task_data: Dict[str, Any]) -> Task:
        """Load a task from a dictionary."""
        self.current_task = Task(**task_data)
        return self.current_task
    
    def load_task_from_file(self, filepath: str) -> Task:
        """Load a task from a JSON file."""
        with open(filepath, 'r') as f:
            task_data = json.load(f)
        return self.load_task(task_data)
    
    def run_reasoning(self) -> List[ReasoningPath]:
        """Run the reasoning process for the current task."""
        if not self.current_task:
            raise ValueError("No task loaded. Call load_task() first.")
        
        logger.info(f"Starting reasoning process for task: {self.current_task.id}")
        
        # Generate multiple reasoning paths
        self.current_paths = self.reasoning_engine.generate_paths(self.current_task)
        
        # Select the best path
        self.best_path = self.reasoning_engine.select_best_path(self.current_paths)
        
        # Log the paths
        self.reasoning_engine._log_paths(self.current_task.id, self.current_paths)
        
        logger.info(f"Completed reasoning. Selected best path: {self.best_path.id} with confidence {self.best_path.confidence:.2f}")
        return self.current_paths
    
    def evaluate_solution(self) -> EvaluationResult:
        """Evaluate the current solution."""
        if not self.current_task or not self.best_path or not self.current_prompt:
            raise ValueError("Task, reasoning path, or prompt not set.")
        
        logger.info("Evaluating solution...")
        
        result = self.evaluator.evaluate_solution(
            task=self.current_task,
            reasoning_path=self.best_path,
            prompt_version=self.current_prompt
        )
        
        logger.info(f"Evaluation complete. Correct: {result.is_correct}, Confidence: {result.confidence:.2f}")
        return result
    
    def optimize_prompt(self, feedback: Optional[Dict[str, float]] = None) -> Optional[PromptVersion]:
        """Optimize the current prompt based on feedback."""
        if not self.current_prompt:
            # Create initial prompt if none exists
            initial_prompt = (
                f"You are an AI assistant solving a {self.current_task.type if self.current_task else 'general'} task. "
                "Provide a clear, step-by-step solution to the problem below. "
                "Think carefully and show your work."
            )
            
            prompt_id = f"task_{self.current_task.id}" if self.current_task else "default_prompt"
            
            # Check if prompt already exists
            try:
                self.current_prompt = self.prompt_optimizer.create_initial_prompt(
                    prompt_id=prompt_id,
                    content=initial_prompt
                )
            except ValueError as e:
                if "already exists" in str(e):
                    # If prompt exists, get the latest version
                    self.current_prompt = max(
                        self.prompt_optimizer.versions[prompt_id],
                        key=lambda x: x.version
                    )
                    logger.info(f"Using existing prompt: {prompt_id} (v{self.current_prompt.version})")
                else:
                    raise
            
            return self.current_prompt
        
        # If no feedback is provided, use default feedback
        if feedback is None:
            feedback = {"accuracy": 0.0, "confidence": 0.0}
        
        # Optimize the prompt
        try:
            self.current_prompt = self.prompt_optimizer.optimize_prompt(
                prompt_id=self.current_prompt.id,
                feedback=feedback
            )
            logger.info(f"Optimized prompt to version {self.current_prompt.version}")
            return self.current_prompt
        except Exception as e:
            logger.error(f"Failed to optimize prompt: {e}")
            return None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics across all evaluations."""
        return self.evaluator.get_performance_metrics()
    
    def run_full_pipeline(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the full pipeline for a task."""
        try:
            # Load task
            self.load_task(task_data)
            
            # Ensure we have a prompt
            if not self.current_prompt:
                self.optimize_prompt()
            
            # Run reasoning
            self.run_reasoning()
            
            # Evaluate
            evaluation = self.evaluate_solution()
            
            # Update prompt based on evaluation
            feedback = {
                "accuracy": 1.0 if evaluation.is_correct else 0.0,
                "confidence": evaluation.confidence
            }
            self.optimize_prompt(feedback)
            
            return {
                "task_id": self.current_task.id if self.current_task else "unknown",
                "success": True,
                "evaluation": evaluation.to_dict(),
                "prompt_version": self.current_prompt.version if self.current_prompt else None
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return {
                "task_id": task_data.get("id", "unknown"),
                "success": False,
                "error": str(e)
            }

def main():
    """Main entry point for the pipeline."""
    # Example usage
    pipeline = Pipeline(num_reasoning_paths=3)
    
    # Example task
    example_task = {
        "id": "example_math_1",
        "type": "math",
        "description": "What is the sum of the first 10 prime numbers?",
        "expected_answer": "129",
        "metadata": {"difficulty": "medium"}
    }
    
    # Run the pipeline
    result = pipeline.run_full_pipeline(example_task)
    print("\nPipeline result:")
    print(json.dumps(result, indent=2))
    
    # Print performance metrics
    metrics = pipeline.get_performance_metrics()
    print("\nPerformance metrics:")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    main()
