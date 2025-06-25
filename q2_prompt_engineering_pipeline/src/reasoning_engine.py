"""Enhanced reasoning engine that integrates with LLM client."""
from typing import List, Dict, Any, Optional
import json
import time
from pathlib import Path
from loguru import logger

from .models import Task, ReasoningPath
from .llm import LLMClient, LLMError
from .config import get_config
from .utils import generate_id, save_json_file, get_logger

class ReasoningEngine:
    """Handles multiple reasoning paths for problem-solving with LLM integration."""
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        num_paths: int = 3,
        log_dir: Optional[Path] = None
    ):
        """Initialize the reasoning engine.
        
        Args:
            llm_client: LLM client to use for generating reasoning paths
            num_paths: Number of parallel reasoning paths to generate
            log_dir: Directory to store reasoning logs
        """
        self.config = get_config()
        self.llm = llm_client or LLMClient()
        self.num_paths = num_paths
        self.logger = get_logger("ReasoningEngine")
        
        # Set up logging directory
        self.log_dir = Path(log_dir) if log_dir else self.config.LOGS_DIR / "reasoning"
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_paths(self, task: Task, num_paths: Optional[int] = None) -> List[ReasoningPath]:
        """Generate multiple reasoning paths for a given task.
        
        Args:
            task: The task to generate reasoning paths for
            num_paths: Number of paths to generate (overrides instance default if provided)
            
        Returns:
            List of ReasoningPath objects
        """
        num_paths = num_paths or self.num_paths
        self.logger.info(f"Generating {num_paths} reasoning paths for task: {task.id}")
        
        try:
            # Generate paths using the LLM
            paths = self.llm.generate_reasoning_paths(task, num_paths=num_paths)
            
            # Log the generated paths
            self._log_paths(task.id, paths)
            
            return paths
            
        except Exception as e:
            self.logger.error(f"Error generating reasoning paths: {e}")
            # Fall back to a simple path if LLM generation fails
            return [self._create_fallback_path(task)]
    
    def _create_fallback_path(self, task: Task) -> ReasoningPath:
        """Create a fallback reasoning path when LLM generation fails."""
        self.logger.warning("Creating fallback reasoning path")
        
        return ReasoningPath(
            id=f"{task.id}_fallback_{int(time.time())}",
            task_id=task.id,
            path=["Fallback reasoning path: Could not generate diverse paths"],
            final_answer=task.expected_answer,  # Use expected answer as fallback
            confidence=0.5,
            metadata={
                "error": "Failed to generate diverse paths",
                "fallback": True
            }
        )
    
    def select_best_path(
        self,
        paths: List[ReasoningPath],
        strategy: str = "confidence",
        **kwargs
    ) -> ReasoningPath:
        """Select the best reasoning path based on the specified strategy.
        
        Args:
            paths: List of ReasoningPath objects
            strategy: Selection strategy ('confidence', 'consensus', 'llm')
            **kwargs: Additional arguments for the selection strategy
            
        Returns:
            The selected ReasoningPath
        """
        if not paths:
            raise ValueError("No paths provided for selection")
            
        self.logger.info(f"Selecting best path using strategy: {strategy}")
        
        if strategy == "confidence":
            return self._select_by_confidence(paths)
        elif strategy == "consensus":
            return self._select_by_consensus(paths, **kwargs)
        elif strategy == "llm":
            return self._select_with_llm(paths, **kwargs)
        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")
    
    def _select_by_confidence(self, paths: List[ReasoningPath]) -> ReasoningPath:
        """Select the path with the highest confidence score."""
        return max(paths, key=lambda x: x.confidence)
    
    def _select_by_consensus(
        self,
        paths: List[ReasoningPath],
        min_agreement: float = 0.7
    ) -> ReasoningPath:
        """Select the most common answer among paths, if there's sufficient agreement."""
        # Group paths by their final answer
        answer_counts: Dict[str, List[ReasoningPath]] = {}
        for path in paths:
            answer = str(path.final_answer).strip().lower()
            if answer not in answer_counts:
                answer_counts[answer] = []
            answer_counts[answer].append(path)
        
        # Find the most common answer
        most_common = max(answer_counts.values(), key=len)
        agreement = len(most_common) / len(paths)
        
        if agreement >= min_agreement:
            # If there's sufficient agreement, return the highest confidence path
            # from the most common answer group
            return max(most_common, key=lambda x: x.confidence)
        else:
            # Otherwise, fall back to confidence-based selection
            self.logger.warning(
                f"Insufficient agreement ({agreement:.2f} < {min_agreement}). "
                "Falling back to confidence-based selection."
            )
            return self._select_by_confidence(paths)
    
    def _select_with_llm(
        self,
        paths: List[ReasoningPath],
        task: Optional[Task] = None,
        criteria: Optional[List[str]] = None
    ) -> ReasoningPath:
        """Use an LLM to select the best reasoning path."""
        if not paths:
            raise ValueError("No paths provided for LLM selection")
            
        if task is None and hasattr(paths[0], 'task_id'):
            # Try to get task from the first path
            task = Task(id=paths[0].task_id, type="unknown", description="", expected_answer="")
        
        criteria = criteria or ["correctness", "logical_consistency", "clarity"]
        
        # Format the paths for the prompt
        paths_info = []
        for i, path in enumerate(paths, 1):
            path_info = {
                "id": path.id,
                "reasoning": "\n".join(path.path),
                "answer": str(path.final_answer),
                "confidence": path.confidence
            }
            paths_info.append(path_info)
        
        # Create the prompt
        system_message = (
            "You are an expert evaluator. Your task is to analyze different reasoning paths "
            "and select the one that best solves the given problem."
        )
        
        prompt = (
            f"Task: {task.description if task else 'Unknown task'}\n\n"
            f"Evaluate the following {len(paths_info)} reasoning paths and select the best one.\n\n"
            f"Evaluation Criteria:\n" + "\n".join(f"- {criterion}" for criterion in criteria) + "\n\n"
            f"Available Paths:\n{json.dumps(paths_info, indent=2)}\n\n"
            "Provide your analysis and select the best path. Format your response as JSON with these fields:\n"
            "- 'best_path_id': ID of the selected path\n"
            "- 'reason': Brief explanation of your choice\n"
            "- 'scores': Object with scores for each criterion"
        )
        
        try:
            # Get the LLM's selection
            response = self.llm.generate_completion(
                prompt=prompt,
                system_message=system_message,
                temperature=0.2,  # Lower temperature for more consistent selections
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            try:
                if '```json' in response:
                    response = response.split('```json')[1].split('```')[0].strip()
                
                selection = json.loads(response)
                best_path_id = selection.get('best_path_id')
                
                # Find the selected path
                selected_path = next((p for p in paths if p.id == best_path_id), None)
                if selected_path is None:
                    self.logger.warning(
                        f"LLM selected invalid path ID: {best_path_id}. "
                        "Falling back to confidence-based selection."
                    )
                    return self._select_by_confidence(paths)
                
                # Update the path's metadata with the selection info
                selected_path.metadata.update({
                    "selection_method": "llm",
                    "selection_reason": selection.get('reason', ''),
                    "selection_scores": selection.get('scores', {})
                })
                
                return selected_path
                
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.error(f"Error parsing LLM selection: {e}")
                self.logger.debug(f"Response content: {response}")
                raise ValueError(f"Failed to parse LLM selection: {e}") from e
                
        except Exception as e:
            self.logger.error(f"Error during LLM-based path selection: {e}")
            # Fall back to confidence-based selection
            return self._select_by_confidence(paths)
    
    def _log_paths(self, task_id: str, paths: List[ReasoningPath]) -> None:
        """Log the generated reasoning paths to a file."""
        if not paths:
            return
            
        timestamp = int(time.time())
        log_data = {
            "task_id": task_id,
            "timestamp": timestamp,
            "paths": [path.dict() for path in paths]
        }
        
        log_file = self.log_dir / f"{task_id}_{timestamp}.json"
        save_json_file(log_data, log_file)
        self.logger.debug(f"Logged {len(paths)} reasoning paths to {log_file}")
    
    def evaluate_paths(
        self,
        task: Task,
        paths: List[ReasoningPath],
        criteria: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluate multiple reasoning paths for a task.
        
        Args:
            task: The task being solved
            paths: List of ReasoningPath objects to evaluate
            criteria: List of evaluation criteria
            
        Returns:
            Dictionary with evaluation results for each path
        """
        results = {}
        
        for path in paths:
            try:
                evaluation = self.llm.evaluate_solution(
                    task=task,
                    solution=path.final_answer,
                    criteria=criteria
                )
                
                # Update the path's confidence based on the evaluation
                path.confidence = evaluation.get('score', 0) / 10  # Scale 0-10 to 0-1
                
                # Store the evaluation results
                results[path.id] = {
                    "score": evaluation.get('score', 0),
                    "is_correct": evaluation.get('is_correct', False),
                    "feedback": evaluation.get('feedback', ''),
                    "scores": evaluation.get('scores', {})
                }
                
            except Exception as e:
                self.logger.error(f"Error evaluating path {path.id}: {e}")
                results[path.id] = {
                    "error": str(e),
                    "score": 0,
                    "is_correct": False
                }
        
        return results
