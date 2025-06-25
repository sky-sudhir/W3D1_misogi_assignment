"""Module for managing tasks in the prompt engineering pipeline."""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import uuid
from datetime import datetime

from loguru import logger
from pydantic import BaseModel, Field, validator

from .models import Task, TaskType
from .utils import load_json_file, save_json_file, generate_id, get_logger
from .config import get_config

class TaskManager:
    """Manages tasks for the prompt engineering pipeline."""
    
    def __init__(self, tasks_dir: Optional[Path] = None):
        """Initialize the task manager.
        
        Args:
            tasks_dir: Directory containing task files
        """
        self.config = get_config()
        self.tasks_dir = Path(tasks_dir) if tasks_dir else self.config.TASKS_DIR
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("TaskManager")
        self._tasks: Dict[str, Task] = {}
        self._load_tasks()
    
    def _load_tasks(self) -> None:
        """Load tasks from the tasks directory."""
        self.logger.info(f"Loading tasks from {self.tasks_dir}")
        
        for task_file in self.tasks_dir.glob("*.json"):
            try:
                task_data = load_json_file(task_file)
                task = Task(**task_data)
                self._tasks[task.id] = task
                self.logger.debug(f"Loaded task: {task.id}")
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                self.logger.error(f"Error loading task from {task_file}: {e}")
    
    def create_task(
        self,
        description: str,                      # Required parameter
        expected_answer: Any,                  # Required parameter
        task_type: Union[str, TaskType] = TaskType.MATH,  # Optional parameter with default
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        save: bool = True
    ) -> Task:
        """Create a new task.
        
        Args:
            task_type: Type of the task (math, code, logic, etc.)
            description: Task description or question
            expected_answer: Expected answer or solution
            task_id: Optional custom task ID
            metadata: Additional task metadata
            save: Whether to save the task to disk
            
        Returns:
            The created Task object
        """
        # Generate a task ID if not provided
        if not task_id:
            prefix = f"{task_type}_" if isinstance(task_type, str) else f"{task_type.value}_"
            task_id = f"{prefix}{generate_id()}"
        
        # Create the task
        task = Task(
            id=task_id,
            type=task_type,
            description=description,
            expected_answer=expected_answer,
            metadata=metadata or {}
        )
        
        # Add to in-memory cache
        self._tasks[task_id] = task
        
        # Save to disk if requested
        if save:
            self.save_task(task)
        
        self.logger.info(f"Created task: {task_id}")
        return task
    
    def save_task(self, task: Task) -> Path:
        """Save a task to disk.
        
        Args:
            task: Task to save
            
        Returns:
            Path to the saved task file
        """
        task_file = self.tasks_dir / f"{task.id}.json"
        task_data = task.dict()
        save_json_file(task_data, task_file)
        self.logger.debug(f"Saved task to {task_file}")
        return task_file
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID.
        
        Args:
            task_id: ID of the task to retrieve
            
        Returns:
            The Task object, or None if not found
        """
        return self._tasks.get(task_id)
    
    def get_all_tasks(self) -> List[Task]:
        """Get all tasks.
        
        Returns:
            List of all Task objects
        """
        return list(self._tasks.values())
    
    def get_tasks_by_type(self, task_type: Union[str, TaskType]) -> List[Task]:
        """Get all tasks of a specific type.
        
        Args:
            task_type: Type of tasks to retrieve
            
        Returns:
            List of matching Task objects
        """
        type_str = task_type.value if isinstance(task_type, TaskType) else task_type
        return [t for t in self._tasks.values() 
                if (isinstance(t.type, str) and t.type.lower() == type_str.lower()) or 
                   (isinstance(t.type, TaskType) and t.type.value.lower() == type_str.lower())]
    
    def delete_task(self, task_id: str) -> bool:
        """Delete a task.
        
        Args:
            task_id: ID of the task to delete
            
        Returns:
            True if the task was deleted, False otherwise
        """
        if task_id not in self._tasks:
            self.logger.warning(f"Task not found: {task_id}")
            return False
        
        # Delete from in-memory cache
        del self._tasks[task_id]
        
        # Delete from disk
        task_file = self.tasks_dir / f"{task_id}.json"
        if task_file.exists():
            task_file.unlink()
            self.logger.info(f"Deleted task: {task_id}")
            return True
        
        return False
    
    def update_task(
        self,
        task_id: str,
        **updates
    ) -> Optional[Task]:
        """Update a task.
        
        Args:
            task_id: ID of the task to update
            **updates: Fields to update
            
        Returns:
            The updated Task object, or None if the task wasn't found
        """
        if task_id not in self._tasks:
            self.logger.warning(f"Task not found: {task_id}")
            return None
        
        # Get the current task data
        task = self._tasks[task_id]
        task_data = task.dict()
        
        # Apply updates
        for key, value in updates.items():
            if key in task_data:
                if key == 'metadata' and value is not None:
                    # Merge metadata dictionaries
                    task_data[key].update(value)
                elif value is not None:
                    task_data[key] = value
        
        # Create updated task
        updated_task = Task(**task_data)
        self._tasks[task_id] = updated_task
        
        # Save to disk
        self.save_task(updated_task)
        self.logger.info(f"Updated task: {task_id}")
        
        return updated_task
    
    def generate_task_batch(
        self,
        task_type: Union[str, TaskType],
        descriptions: List[str],
        expected_answers: List[Any],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        prefix: str = "batch_"
    ) -> List[Task]:
        """Generate multiple tasks at once.
        
        Args:
            task_type: Type of tasks to create
            descriptions: List of task descriptions
            expected_answers: List of expected answers (must match descriptions length)
            metadata_list: Optional list of metadata dictionaries
            prefix: Prefix for generated task IDs
            
        Returns:
            List of created Task objects
        """
        if len(descriptions) != len(expected_answers):
            raise ValueError("Length of descriptions must match length of expected_answers")
        
        if metadata_list is not None and len(metadata_list) != len(descriptions):
            raise ValueError("Length of metadata_list must match length of descriptions")
        
        tasks = []
        for i, (desc, ans) in enumerate(zip(descriptions, expected_answers)):
            metadata = metadata_list[i] if metadata_list else None
            task_id = f"{prefix}{i:03d}"
            
            task = self.create_task(
                task_type=task_type,
                description=desc,
                expected_answer=ans,
                task_id=task_id,
                metadata=metadata,
                save=True
            )
            tasks.append(task)
        
        self.logger.info(f"Generated {len(tasks)} tasks of type {task_type}")
        return tasks
