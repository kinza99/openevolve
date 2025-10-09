system_prompt = (
    f'You are a helpful assistant that can generate testing for a given task.\n'
    'You will be given a task and the environment building script. You need to generate a testing script that can be used to test agent\'s solution.\n'
)
qa_example = """#!/bin/bash

./folder1/folder2/folder3/folder4/folder5/echo-love 2>&1
"""
qa_instruction = (
    'Please give me the testing script for this task to get the ground truth of the task.\n'
    'The content you generate should be able to serve as the content of an executble script. The execution result of the testing script should be just the clean ground truth of the task.\n'
    'Please encapsulate your final testing script (script content ONLY) within <testing> and </testing>.\n'
    f'For example: The testing script is <testing> {qa_example} </testing>.\n'
    '# Problem \n'
    '{description}\n\n'
    '# Environment Building Script \n'
    '{init}\n\n'
)

execution_example = """#!/bin/bash

# 执行第一个命令并获取输出
output1=$(source ~/.bashrc && echo-love 2>&1)

# 执行第二个命令并获取输出  
output2=$(./folder1/folder2/folder3/folder4/folder5/echo-love 2>&1)

# 比较输出结果
if [ "$output1" = "$output2" ]; then
    echo "true"
else
    echo "false"
fi"""

execution_instruction = (
    'Please give me the testing script for this task to judge the correctness of the agent\'s execution solution.\n'
    'The content you generate should be able to serve as the content of an executble script. The execution result of the testing script should be just the boolean value of the correctness of the agent\'s execution solution.\n'
    'Testing should get the result or effect of the agent\'s execution solution first and then get the ground truth of the task.\n'
    'Finally, testing should compare the result with the ground truth and output the boolean value of the correctness of the agent\'s execution solution.\n'
    'Please encapsulate your final testing script (script content ONLY) within <testing> and </testing>.\n'
    f'For example: The testing script is <testing> {execution_example} </testing>.\n'
    '# Problem \n'
    '{description}\n\n'
    '# Environment Building Script \n'
    '{init}\n\n'
)