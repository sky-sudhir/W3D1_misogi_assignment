from typing import Dict, List, Optional
from pathlib import Path
from loguru import logger
import json
import random

from .models import PromptVersion, Task, ReasoningPath

class PromptOptimizer:
    """Handles optimization of prompts based on performance feedback."""
    
    def __init__(self, prompts_dir: str = "./prompts"):
        """Initialize the prompt optimizer.
        
        Args:
            prompts_dir: Directory to store prompt versions
        """
        self.prompts_dir = Path(prompts_dir)
        self.prompts_dir.mkdir(exist_ok=True)
        self.versions: Dict[str, List[PromptVersion]] = {}
        self._load_existing_prompts()
    
    def _load_existing_prompts(self):
        """Load existing prompts from the prompts directory."""
        print(f"Looking for prompt files in: {self.prompts_dir.resolve()}")
        for prompt_file in self.prompts_dir.glob("*.json"):
            try:
                print(f"Loading prompt from: {prompt_file.resolve()}")
                with open(prompt_file, 'r') as f:
                    prompt_data = json.load(f)
                prompt_version = PromptVersion(**prompt_data)
                
                if prompt_version.id not in self.versions:
                    self.versions[prompt_version.id] = []
                self.versions[prompt_version.id].append(prompt_version)
                
            except Exception as e:
                logger.error(f"Error loading prompt from {prompt_file.resolve()}: {e}")
    
    def create_initial_prompt(self, prompt_id: str, content: str) -> PromptVersion:
        """Create an initial version of a prompt.
        
        Args:
            prompt_id: Unique identifier for the prompt
            content: The prompt content
            
        Returns:
            The created PromptVersion
        """
        if prompt_id in self.versions:
            raise ValueError(f"Prompt with id {prompt_id} already exists")
            
        version = PromptVersion(
            id=prompt_id,
            content=content,
            version=1,
            performance_metrics={"initial": 0.0}
        )
        
        self.versions[prompt_id] = [version]
        self._save_prompt_version(version)
        return version
    
    def optimize_prompt(
        self,
        prompt_id: str,
        feedback: Dict[str, float],
        strategy: str = "modify"
    ) -> Optional[PromptVersion]:
        """Optimize a prompt based on feedback.
        
        Args:
            prompt_id: ID of the prompt to optimize
            feedback: Performance metrics from the last run
            strategy: Optimization strategy ('modify' or 'mutate')
            
        Returns:
            New PromptVersion if optimization was successful, None otherwise
        """
        if prompt_id not in self.versions:
            raise ValueError(f"No prompt found with id {prompt_id}")
            
        current_versions = self.versions[prompt_id]
        latest_version = max(current_versions, key=lambda x: x.version)
        
        # In a real implementation, this would use an LLM to optimize the prompt
        # For now, we'll just create a simple modification
        if strategy == "modify":
            new_content = self._modify_prompt(latest_version.content, feedback)
        elif strategy == "mutate":
            new_content = self._mutate_prompt(latest_version.content)
        else:
            raise ValueError(f"Unknown optimization strategy: {strategy}")
        
        # Create new version
        new_version = PromptVersion(
            id=prompt_id,
            content=new_content,
            version=latest_version.version + 1,
            parent_version=latest_version.version,
            performance_metrics=feedback
        )
        
        self.versions[prompt_id].append(new_version)
        self._save_prompt_version(new_version)
        return new_version
    
    def _modify_prompt(self, content: str, feedback: Dict[str, float]) -> str:
        """Modify prompt based on feedback."""
        # This is a placeholder - in a real implementation, this would use an LLM
        # to analyze feedback and modify the prompt accordingly
        modifications = [
            "\n\nPlease think step by step and explain your reasoning.",
            "\n\nConsider multiple approaches before providing an answer.",
            "\n\nDouble-check your work for accuracy and completeness."
        ]
        
        return content + random.choice(modifications)
    
    def _mutate_prompt(self, content: str) -> str:
        """Create a mutated version of the prompt."""
        # Simple mutation - in a real implementation, this would be more sophisticated
        mutations = [
            lambda s: s.replace("Solve", "Carefully analyze and solve"),
            lambda s: s + "\n\nShow your work and explain your reasoning.",
            lambda s: s + "\n\nConsider edge cases and verify your solution."
        ]
        
        return random.choice(mutations)(content)
    
    def _save_prompt_version(self, version: PromptVersion):
        """Save a prompt version to disk."""
        filename = f"{version.id}_v{version.version}.json"
        filepath = self.prompts_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(version.dict(), f, indent=2)
    
    def get_latest_version(self, prompt_id: str) -> Optional[PromptVersion]:
        """Get the latest version of a prompt."""
        if prompt_id not in self.versions or not self.versions[prompt_id]:
            return None
        return max(self.versions[prompt_id], key=lambda x: x.version)
