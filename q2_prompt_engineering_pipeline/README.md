# Prompt Engineering Pipeline

A sophisticated system for solving complex problems using multiple reasoning paths and self-improving prompts. This implementation follows the Tree of Thought (ToT) and Self-Consistency approaches, with automatic prompt optimization.

## Features

- **Multiple Reasoning Paths**: Generate and evaluate multiple approaches to problem-solving
- **Self-Improving Prompts**: Automatically optimize prompts based on performance
- **Task Management**: Create, update, and manage different types of tasks
- **Evaluation Framework**: Comprehensive metrics for evaluating solution quality
- **Extensible Architecture**: Easy to integrate with different LLM providers

## Project Structure

```
prompt_engineering_pipeline/
├── src/                    # Source code
│   ├── __init__.py
│   ├── config.py          # Configuration settings
│   ├── evaluation.py      # Evaluation framework
│   ├── llm.py            # LLM client and utilities
│   ├── main.py           # Main pipeline implementation
│   ├── models.py         # Pydantic models
│   ├── prompt_optimizer.py # Prompt optimization logic
│   ├── reasoning_engine.py # Reasoning path generation and selection
│   ├── task_manager.py    # Task management
│   └── utils.py          # Utility functions
├── tasks/                 # Task definitions
├── prompts/              # Prompt templates and versions
├── logs/                 # Log files
├── evaluation/           # Evaluation results
├── tests/                # Unit tests
├── .env.example         # Example environment variables
├── .gitignore
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd prompt_engineering_pipeline
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Unix or MacOS:
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

## Usage

### Running the Pipeline

```python
from src.main import Pipeline
from src.models import Task

# Create a pipeline instance
pipeline = Pipeline(num_reasoning_paths=3)

# Define a task
task = Task(
    id="example_math_1",
    type="math",
    description="What is the sum of the first 10 prime numbers?",
    expected_answer="129"
)

# Run the full pipeline
result = pipeline.run_full_pipeline(task.dict())
print(f"Pipeline result: {result}")
```

### Task Management

```python
from src.task_manager import TaskManager

# Initialize task manager
task_manager = TaskManager()

# Create a new task
task = task_manager.create_task(
    task_type="math",
    description="What is the square root of 144?",
    expected_answer="12"
)

# Get all tasks
tasks = task_manager.get_all_tasks()

# Get tasks by type
math_tasks = task_manager.get_tasks_by_type("math")
```

### Customizing the Pipeline

```python
from src.llm import LLMClient
from src.reasoning_engine import ReasoningEngine
from src.prompt_optimizer import PromptOptimizer

# Customize the pipeline components
llm_client = LLMClient(model="gpt-4")
reasoning_engine = ReasoningEngine(
    llm_client=llm_client,
    num_paths=5
)
prompt_optimizer = PromptOptimizer()

# Use custom components in the pipeline
pipeline = Pipeline(
    reasoning_engine=reasoning_engine,
    prompt_optimizer=prompt_optimizer
)
```

## Configuration

Configuration is managed through environment variables in the `.env` file:

```ini
# LLM Settings
OPENAI_API_KEY=your-api-key-here
DEFAULT_MODEL=gpt-4
TEMPERATURE=0.7
MAX_TOKENS=2048

# Paths
LOGS_DIR=logs
PROMPTS_DIR=prompts
TASKS_DIR=tasks
EVALUATION_DIR=evaluation

# Logging
LOG_LEVEL=INFO
```

## Running Tests

```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# With coverage report
pytest --cov=src tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by Tree of Thoughts (ToT) and Self-Consistency approaches
- Built with Python and the OpenAI API
- Thanks to all contributors who have helped improve this project Engineering Pipeline

A smart system that solves complex problems using multiple reasoning paths and self-improving prompts.

## Project Structure

- `tasks/`: Contains problem statements and their expected solutions
- `src/`: Main source code for the pipeline
- `prompts/`: Original and improved prompt versions
- `logs/`: Logs of reasoning paths and system operations
- `evaluation/`: Performance metrics and analysis
- `.venv/`: Python virtual environment (gitignored)

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the pipeline:

```bash
python src/main.py
```

## Features

- Multiple reasoning paths for problem-solving
- Self-improving prompt optimization
- Performance tracking and evaluation
- Logging and analysis tools
