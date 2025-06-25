#!/usr/bin/env python3
"""
Example script to run the prompt engineering pipeline.
"""
import json
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))

from main import Pipeline

def main():
    """Run the pipeline with an example task."""
    # Initialize the pipeline
    pipeline = Pipeline(num_reasoning_paths=3)
    
    # Load the example task
    task_path = Path("tasks/example_math_task.json")
    with open(task_path, 'r') as f:
        task_data = json.load(f)
    
    print(f"Running pipeline for task: {task_data['id']}")
    print(f"Description: {task_data['description']}")
    print("-" * 50)
    
    # Run the full pipeline
    result = pipeline.run_full_pipeline(task_data)
    
    print("\nPipeline execution complete!")
    print("-" * 50)
    print("Results:")
    print(f"Task ID: {result['task_id']}")
    print(f"Success: {result['success']}")
    
    if result['success']:
        eval_result = result['evaluation']
        print(f"Correct: {eval_result['is_correct']}")
        print(f"Confidence: {eval_result['confidence']:.2f}")
        print(f"Prompt Version: {result['prompt_version']}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    # Print performance metrics
    metrics = pipeline.get_performance_metrics()
    print("\nPerformance Metrics:")
    print(f"Total Tasks: {metrics.get('total_tasks', 0)}")
    print(f"Accuracy: {metrics.get('accuracy', 0.0):.2%}")
    print(f"Average Confidence: {metrics.get('average_confidence', 0.0):.2f}")

if __name__ == "__main__":
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    main()
