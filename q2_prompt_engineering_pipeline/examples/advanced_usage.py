"""
Advanced usage example for the Prompt Engineering Pipeline.

This script demonstrates:
1. Custom task creation and management
2. Advanced pipeline configuration
3. Prompt optimization and versioning
4. Performance evaluation and metrics
"""
import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.main import Pipeline
from src.task_manager import TaskManager
from src.models import Task, TaskType, ReasoningPath
from src.llm import LLMClient
from src.reasoning_engine import ReasoningEngine
from src.prompt_optimizer import PromptOptimizer
from src.evaluation import Evaluator
from src.config import get_config

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedPipelineDemo:
    """Demonstrates advanced usage of the prompt engineering pipeline."""
    
    def __init__(self):
        """Initialize the demo with default configuration."""
        self.config = get_config()
        self.setup_environment()
        
        # Initialize components
        self.task_manager = TaskManager()
        self.llm_client = LLMClient()
        self.reasoning_engine = ReasoningEngine(llm_client=self.llm_client, num_paths=3)
        self.prompt_optimizer = PromptOptimizer()
        self.evaluator = Evaluator()
        
        # Initialize the pipeline with custom components
        self.pipeline = Pipeline(
            reasoning_engine=self.reasoning_engine,
            prompt_optimizer=self.prompt_optimizer,
            evaluator=self.evaluator
        )
        
        # Results storage
        self.results = []
    
    def setup_environment(self):
        """Ensure all required directories exist."""
        for directory in ["logs", "prompts", "tasks", "evaluation", "examples/output"]:
            Path(directory).mkdir(exist_ok=True, parents=True)
    
    def load_or_create_tasks(self) -> List[Task]:
        """Load existing tasks or create sample tasks if none exist."""
        tasks = self.task_manager.get_all_tasks()
        
        if not tasks:
            logger.info("No tasks found. Creating sample tasks...")
            tasks = self._create_sample_tasks()
        else:
            logger.info(f"Loaded {len(tasks)} existing tasks")
        
        return tasks
    
    def _create_sample_tasks(self) -> List[Task]:
        """Create a diverse set of sample tasks."""
        task_definitions = [
            # Math tasks
            {
                "type": TaskType.MATH,
                "description": "What is the sum of the first 10 prime numbers?",
                "expected_answer": "129",
                "metadata": {"difficulty": "easy", "category": "number theory"}
            },
            {
                "type": TaskType.MATH,
                "description": "Solve for x: 3x + 10 = 5x - 2",
                "expected_answer": "6",
                "metadata": {"difficulty": "easy", "category": "algebra"}
            },
            
            # Code tasks
            {
                "type": TaskType.CODE,
                "description": "Write a Python function to calculate the factorial of a number.",
                "expected_answer": "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)",
                "metadata": {"difficulty": "easy", "language": "python"}
            },
            {
                "type": TaskType.CODE,
                "description": "Write a function to find the longest common prefix among an array of strings.",
                "expected_answer": "def longest_common_prefix(strs):\n    if not strs:\n        return ''\n    prefix = strs[0]\n    for s in strs[1:]:\n        while not s.startswith(prefix):\n            prefix = prefix[:-1]\n            if not prefix:\n                return ''\n    return prefix",
                "metadata": {"difficulty": "medium", "language": "python"}
            },
            
            # Logic tasks
            {
                "type": TaskType.LOGIC,
                "description": "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops definitely Lazzies?",
                "expected_answer": "Yes, all Bloops are definitely Lazzies.",
                "metadata": {"difficulty": "medium", "category": "logic"}
            },
            {
                "type": TaskType.LOGIC,
                "description": "A man lives on the 10th floor of a building. Every day he takes the elevator to go down to the ground floor to go to work. When he returns he takes the elevator to the 7th floor and walks up the stairs to reach his apartment on the 10th floor. He hates walking so why does he do it?",
                "expected_answer": "The man is a dwarf and cannot reach the button for the 10th floor. He can reach the 7th floor button and walks up the remaining floors.",
                "metadata": {"difficulty": "hard", "category": "riddle"}
            },
            
            # Language tasks
            {
                "type": TaskType.LANGUAGE,
                "description": "Correct the grammar in this sentence: 'me and my friend goes to the park yesterday'",
                "expected_answer": "My friend and I went to the park yesterday.",
                "metadata": {"difficulty": "easy", "language": "english"}
            },
            {
                "type": TaskType.LANGUAGE,
                "description": "Translate to French: 'The quick brown fox jumps over the lazy dog.'",
                "expected_answer": "Le rapide renard brun saute par-dessus le chien paresseux.",
                "metadata": {"difficulty": "medium", "language": "french"}
            }
        ]
        
        tasks = []
        for i, task_data in enumerate(task_definitions, 1):
            task = self.task_manager.create_task(
                task_id=f"task_{i:03d}",
                **task_data
            )
            tasks.append(task)
            logger.info(f"Created task: {task.id} - {task.description[:50]}...")
        
        return tasks
    
    def run_pipeline(self, task: Task, max_retries: int = 3) -> Dict[str, Any]:
        """Run the pipeline for a single task with retries."""
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing task: {task.id}")
        logger.info(f"Type: {task.type}")
        logger.info(f"Description: {task.description}")
        
        result = None
        for attempt in range(max_retries):
            try:
                # Run the pipeline
                logger.info(f"\nPipeline attempt {attempt + 1}/{max_retries}")
                result = self.pipeline.run_full_pipeline(task.dict())
                
                # Log the result
                logger.info(f"Pipeline completed: {result['success']}")
                if result['success']:
                    eval_result = result['evaluation']
                    logger.info(f"Is correct: {eval_result['is_correct']}")
                    logger.info(f"Confidence: {eval_result['confidence']:.2f}")
                
                # Store the result
                self.results.append({
                    "task_id": task.id,
                    "task_type": task.type,
                    "success": result['success'],
                    "is_correct": eval_result['is_correct'] if result['success'] else False,
                    "confidence": eval_result['confidence'] if result['success'] else 0.0,
                    "prompt_version": result.get('prompt_version', 1),
                    "timestamp": datetime.now().isoformat(),
                    "attempt": attempt + 1
                })
                
                # Save results after each task
                self.save_results()
                
                # Break on success
                if result['success']:
                    break
                    
            except Exception as e:
                logger.error(f"Error in pipeline attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:  # Last attempt
                    logger.error(f"Failed to process task {task.id} after {max_retries} attempts")
                    self.results.append({
                        "task_id": task.id,
                        "task_type": task.type,
                        "success": False,
                        "is_correct": False,
                        "confidence": 0.0,
                        "prompt_version": 0,
                        "timestamp": datetime.now().isoformat(),
                        "error": str(e),
                        "attempt": attempt + 1
                    })
                    self.save_results()
        
        return result or {"success": False, "error": "All attempts failed"}
    
    def save_results(self):
        """Save results to a CSV file."""
        if not self.results:
            return
            
        df = pd.DataFrame(self.results)
        output_dir = Path("examples/output")
        output_dir.mkdir(exist_ok=True)
        
        # Save full results
        df.to_csv(output_dir / "pipeline_results.csv", index=False)
        
        # Save summary
        if len(df) > 0:
            summary = {
                "total_tasks": len(df['task_id'].unique()),
                "success_rate": df['success'].mean(),
                "accuracy": df[df['success']]['is_correct'].mean() if df['success'].any() else 0,
                "avg_confidence": df[df['success']]['confidence'].mean() if df['success'].any() else 0,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(output_dir / "summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
    
    def analyze_results(self):
        """Analyze and visualize the results."""
        if not self.results:
            logger.warning("No results to analyze")
            return
        
        df = pd.DataFrame(self.results)
        
        # Basic statistics
        print("\n=== Results Summary ===")
        print(f"Total tasks: {len(df['task_id'].unique())}")
        print(f"Success rate: {df['success'].mean():.1%}")
        print(f"Accuracy (successful tasks): {df[df['success']]['is_correct'].mean():.1%}")
        print(f"Average confidence: {df[df['success']]['confidence'].mean():.2f}")
        
        # Performance by task type
        if 'task_type' in df.columns and not df.empty:
            print("\nPerformance by task type:")
            type_stats = df[df['success']].groupby('task_type').agg({
                'is_correct': 'mean',
                'confidence': 'mean',
                'task_id': 'count'
            }).rename(columns={'task_id': 'count'})
            print(type_stats)
            
            # Plot performance by task type
            if not type_stats.empty:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                # Accuracy by task type
                type_stats['is_correct'].sort_values().plot(
                    kind='barh', ax=ax1, title='Accuracy by Task Type'
                )
                ax1.set_xlabel('Accuracy')
                
                # Confidence by task type
                type_stats['confidence'].sort_values().plot(
                    kind='barh', ax=ax2, title='Confidence by Task Type', color='orange'
                )
                ax2.set_xlabel('Average Confidence')
                
                plt.tight_layout()
                plt.savefig('examples/output/performance_by_type.png')
                print("\nSaved performance visualization to examples/output/performance_by_type.png")
    
    def run_demo(self):
        """Run the complete demo."""
        print("\n=== Advanced Prompt Engineering Pipeline Demo ===\n")
        
        try:
            # Load or create tasks
            tasks = self.load_or_create_tasks()
            
            # Process each task
            for task in tasks:
                self.run_pipeline(task)
            
            # Analyze and display results
            self.analyze_results()
            
            print("\n=== Demo Complete ===")
            print("Results have been saved to the 'examples/output' directory.")
            
        except Exception as e:
            logger.exception("An error occurred during the demo")
            raise

def main():
    """Run the advanced demo."""
    # Load environment variables
    load_dotenv()
    
    # Run the demo
    demo = AdvancedPipelineDemo()
    demo.run_demo()

if __name__ == "__main__":
    main()
