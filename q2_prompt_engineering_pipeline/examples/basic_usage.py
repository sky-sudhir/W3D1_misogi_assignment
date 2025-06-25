"""
Basic usage example for the Prompt Engineering Pipeline.

This script demonstrates how to:
1. Create and manage tasks
2. Run the full pipeline
3. Evaluate results
4. Optimize prompts
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.main import Pipeline
from src.task_manager import TaskManager
from src.models import TaskType

def setup_environment():
    """Load environment variables and ensure required directories exist."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Ensure required directories exist
    for directory in ["logs", "prompts", "tasks", "evaluation"]:
        Path(directory).mkdir(exist_ok=True)

def create_sample_tasks(task_manager: TaskManager):
    """Create some sample tasks for demonstration."""
    tasks = [
        {
            "task_type": TaskType.MATH,
            "description": "What is the sum of the first 10 prime numbers?",
            "expected_answer": "129",
            "metadata": {"difficulty": "easy", "category": "number theory"}
        },
        {
            "task_type": TaskType.CODE,
            "description": "Write a Python function to check if a string is a palindrome.",
            "expected_answer": "def is_palindrome(s):\n    return s == s[::-1]",
            "metadata": {"difficulty": "easy", "language": "python"}
        },
        {
            "task_type": TaskType.LOGIC,
            "description": "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops definitely Lazzies?",
            "expected_answer": "Yes, all Bloops are definitely Lazzies.",
            "metadata": {"difficulty": "medium", "category": "logic"}
        }
    ]
    
    created_tasks = []
    for i, task_data in enumerate(tasks, 1):
        task = task_manager.create_task(
            task_id=f"example_{i}",
            **task_data
        )
        created_tasks.append(task)
        print(f"Created task: {task.id} - {task.description[:50]}...")
    
    return created_tasks

def run_pipeline_demo():
    """Run a demo of the pipeline with sample tasks."""
    print("\n=== Prompt Engineering Pipeline Demo ===\n")
    
    # Initialize components
    task_manager = TaskManager()
    pipeline = Pipeline(num_reasoning_paths=3)
    
    # Create sample tasks
    print("\nCreating sample tasks...")
    tasks = create_sample_tasks(task_manager)
    
    # Process each task
    for task in tasks:
        print(f"\n{'='*50}")
        print(f"Processing task: {task.id}")
        print(f"Type: {task.type}")
        print(f"Description: {task.description}")
        
        try:
            # Run the full pipeline
            print("\nRunning pipeline...")
            result = pipeline.run_full_pipeline(task.dict())
            
            # Print results
            print("\nPipeline result:")
            print(f"Success: {result['success']}")
            
            if result['success']:
                eval_result = result['evaluation']
                print(f"Is correct: {eval_result['is_correct']}")
                print(f"Confidence: {eval_result['confidence']:.2f}")
                print(f"Prompt version: {result['prompt_version']}")
            
        except Exception as e:
            print(f"\nError processing task {task.id}: {str(e)}")
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    setup_environment()
    run_pipeline_demo()
