system_prompt = """You are a helpful assistant and your job is generating competitive programming problems.
"""

func_prompt = """Your job is generating a competitive programming problem following the given example. The problem should can be solved by a python code. A problem should contains 4 parts:
1. Description: The problem natural language description, some simple unit test examples could be in the description.
2. Title: The title of the problem, it should be a short and clear title only contanins few words.
3. Starter_Code: The starter code of the problem, it is the starter part of answer python code and contains the function signature. Note, the starter code must be in the form of a signature of a function to be implemented under the Solution class, like:
```python
class Solution:
    def {{function_name}}(self, {{parameters}}):
        ...
```
4. Function_name: The name of the function or class in the starter code. This function or class will be executed during the evaluation.

Example:
### Description
{example_description}

### Title
{example_title}

### Starter_Code
{example_starter_code}

### Function_name
{example_function_name}

Now generate another problem following the given example and example's format.
"""

std_prompt = """Your job is generating a standard competitive programming problem following the given example. The problem should can be solved by a python code. A problem should contains 2 parts:
1. Description: The problem natural language description, some simple unit test examples could be in the description.
2. Title: The title of the problem, it should be a short and clear title only contanins few words.

Example:
### Description
{example_description}

### Title
{example_title}
Now generate another problem following the given example and example's format.
"""