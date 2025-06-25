from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum

class TaskType(str, Enum):
    MATH = "math"
    CODE = "code"
    LOGIC = "logic"
    LANGUAGE = "language"
    OTHER = "other"

class Task(BaseModel):
    """Represents a problem to be solved."""
    id: str
    type: TaskType
    description: str
    expected_answer: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ReasoningPath(BaseModel):
    """Represents a single reasoning path for solving a task."""
    id: str
    task_id: str
    path: List[str]  # List of reasoning steps
    final_answer: Any
    confidence: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

class PromptVersion(BaseModel):
    """Represents a version of a prompt with its performance metrics."""
    id: str
    content: str
    version: int
    parent_version: Optional[int] = None
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
