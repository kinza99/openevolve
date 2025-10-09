system_prompt = (
    'You are a helpful assistant that can recognize the type of the given query.\n'
    'You will be given a query. You need to recognize the type of the query and output the type of the query.\n'
)

type_instruction = """Please classify the type of the following query. The query can either be a QA (Question and Answer) type or an EXEC (Execution) type.

1. QA: A query that asks a question and expects a direct, clear answer, like getting the number of something, or getting the content of a file. The agent needs to provide specific information based on the query.
   - Example: "How many hidden files are in /usr?" or "What is the total number of files in ./project_directory?" or "Calculate the sum of the numbers in the file /root/matrix.txt." or "Output the content of the file /root/matrix.txt."

2. EXEC: A query that requests a task or action that the agent must perform. It usually involves executing commands, modifying configurations, or performing system-related operations.
   - Example: "Add the directory containing echo-love to the PATH" or "Create a function that evaluates mathematical expressions when the 'calc' command is used."

### Guidelines:
- For QA queries, the agent is expected to provide factual answers based on the query's context.
- For EXEC queries, the agent is expected to execute a task or series of actions to fulfill the request.

### Task:
Encapsulate the type of the query as either <type> QA </type> or <type> EXEC </type>. Your answer should only include the type, without additional information.

### Example:
- Query: "How many lines are there in /root/matrix.txt?"
  - Type: <type> QA </type>

- Query: "Create a new file called /root/matrix.txt"
  - Type: <type> EXEC </type>

---

### Query:
{query}
"""