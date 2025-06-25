from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
from pathlib import Path
from loguru import logger

from .models import Task, ReasoningPath, PromptVersion

@dataclass
class EvaluationResult:
    """Stores evaluation results for a single task."""
    task_id: str
    is_correct: bool
    confidence: float
    metrics: Dict[str, float]
    reasoning_path_id: str
    prompt_version: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "is_correct": self.is_correct,
            "confidence": self.confidence,
            "metrics": self.metrics,
            "reasoning_path_id": self.reasoning_path_id,
            "prompt_version": self.prompt_version
        }

class Evaluator:
    """Handles evaluation of the system's performance."""
    
    def __init__(self, eval_dir: str = "../evaluation"):
        """Initialize the evaluator.
        
        Args:
            eval_dir: Directory to store evaluation results
        """
        self.eval_dir = Path(eval_dir)
        self.eval_dir.mkdir(exist_ok=True)
    
    def evaluate_solution(
        self,
        task: Task,
        reasoning_path: ReasoningPath,
        prompt_version: PromptVersion
    ) -> EvaluationResult:
        """Evaluate a solution to a task.
        
        Args:
            task: The original task
            reasoning_path: The reasoning path used to solve the task
            prompt_version: The prompt version used
            
        Returns:
            EvaluationResult with the evaluation
        """
        # In a real implementation, this would use more sophisticated evaluation
        # For now, we'll use a simple comparison of expected vs actual answer
        is_correct = (
            str(reasoning_path.final_answer).strip().lower() == 
            str(task.expected_answer).strip().lower()
        )
        
        # Calculate some basic metrics
        metrics = {
            "confidence": reasoning_path.confidence,
            "path_length": len(reasoning_path.path),
            "is_correct": float(is_correct)
        }
        
        result = EvaluationResult(
            task_id=task.id,
            is_correct=is_correct,
            confidence=reasoning_path.confidence,
            metrics=metrics,
            reasoning_path_id=reasoning_path.id,
            prompt_version=f"{prompt_version.id}_v{prompt_version.version}"
        )
        
        self._log_evaluation(result, task, reasoning_path, prompt_version)
        return result
    
    def _log_evaluation(
        self,
        result: EvaluationResult,
        task: Task,
        reasoning_path: ReasoningPath,
        prompt_version: PromptVersion
    ):
        """Log evaluation results to a file."""
        log_entry = {
            "timestamp": self._get_timestamp(),
            "evaluation": result.to_dict(),
            "task": {
                "id": task.id,
                "type": task.type,
                "description": task.description[:200] + "..." if len(task.description) > 200 else task.description
            },
            "reasoning_path": {
                "id": reasoning_path.id,
                "steps": len(reasoning_path.path),
                "confidence": reasoning_path.confidence
            },
            "prompt_version": {
                "id": prompt_version.id,
                "version": prompt_version.version
            }
        }
        
        log_file = self.eval_dir / f"evaluation_{task.id}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate and return performance metrics across all evaluations."""
        metrics = {
            "total_tasks": 0,
            "correct_answers": 0,
            "total_confidence": 0.0,
            "by_task_type": {}
        }
        
        # Process all evaluation files
        for eval_file in self.eval_dir.glob("evaluation_*.jsonl"):
            with open(eval_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        eval_data = entry["evaluation"]
                        task_type = entry["task"].get("type", "unknown")
                        
                        # Update overall metrics
                        metrics["total_tasks"] += 1
                        if eval_data["is_correct"]:
                            metrics["correct_answers"] += 1
                        metrics["total_confidence"] += eval_data["confidence"]
                        
                        # Update per-task-type metrics
                        if task_type not in metrics["by_task_type"]:
                            metrics["by_task_type"][task_type] = {
                                "total": 0,
                                "correct": 0,
                                "total_confidence": 0.0
                            }
                        
                        metrics["by_task_type"][task_type]["total"] += 1
                        if eval_data["is_correct"]:
                            metrics["by_task_type"][task_type]["correct"] += 1
                        metrics["by_task_type"][task_type]["total_confidence"] += eval_data["confidence"]
                        
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Error processing evaluation entry: {e}")
        
        # Calculate averages
        if metrics["total_tasks"] > 0:
            metrics["accuracy"] = metrics["correct_answers"] / metrics["total_tasks"]
            metrics["average_confidence"] = metrics["total_confidence"] / metrics["total_tasks"]
            
            # Calculate per-task-type averages
            for task_type in metrics["by_task_type"]:
                task_metrics = metrics["by_task_type"][task_type]
                if task_metrics["total"] > 0:
                    task_metrics["accuracy"] = task_metrics["correct"] / task_metrics["total"]
                    task_metrics["average_confidence"] = task_metrics["total_confidence"] / task_metrics["total"]
        
        return metrics
    
    @staticmethod
    def _get_timestamp() -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"
