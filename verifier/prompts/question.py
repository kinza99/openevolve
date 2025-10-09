system_prompt = """You are a helpful assistant and your job is generating realistic task for the given example.
"""

user_prompt = """Your job is generating a realistic task following the given example. The task is a real OS task, which could be solved by a solution script. The task contains 4 parts:

1. Query: The task natural language description.
2. Init: The environment initialization script for the task, it should be a script that can be executed to initialize the environment for the task.
3. Type: A problem can only belong to two types, QA and EXEC. 
    - QA: This type of task asks a question and expects a direct, clear answer, like getting the number of something, or getting the content of a file. The agent needs to provide specific information based on the query.
    - EXEC: This type of task requests a task or action that the agent must perform. It usually involves executing commands, modifying configurations, or performing system-related operations.
4. Comparison_method: The method to check the agent's answer. It can only be "check/integer-match.py" or "check/size-match.py" or "check/string-match.py".

Please generate the task description and the environment initialization script following the given example and example's format. The task description should be a real OS task, which could be solved by the a solution script. The environment initialization script should be a script that can initialize the environment for the task.

Example:
### Query:
{example_query}

### Init:
{example_init}

### Type:
{example_type}

### Comparison_method:
{example_comparison_method}

Now generate another task following the given example and example's format.
"""