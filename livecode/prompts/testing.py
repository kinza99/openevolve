system_prompt = (
    f'You are a expert in wirting unit tests. You will be given a problem of a programming task and you need to generate as comprehensive and accurate unit tests as possible.'
)

assistant_prompt = """I will give you a natural language description of a programming problem and you need to generate unit tests that cover all edge cases of the problem.
Good unit tests should cover all inputs that are mentioned in the problem description, as well as any unique edge cases that might not be obvious to the user.
The description of the problem might contains some simple unit test examples, but they are not enough to verify the correctness of the solution. So you should refer to them to generate more comprehensive unit tests.
The format of the unit tests should be a strict json dict which can be loaded by json.loads() in python. And the structure of the json dict should be a list of dicts, like following:

```json
[
    {
        "input": "input1",
        "output": "output1"
    },
    {
        "input": "input2",
        "output": "output2"
    }
    ...
]
```
The type of the input or output value should follow the type of the input or output value in the problem description strictly.
The unit tests you generated should contain all the edge cases that are mentioned in the problem description, and they also should follow the json format of the example unit tests.
The unit tests you generated should follow the constraints of the problem description strictly.
Sometimes the input of unit test may be very long, you can use a simple python code to generate the input like Problem 3, the python code must can be process by the eval() function in python.
You should try to maximize the quality and coverage of the unit tests, here are some examples of good unit tests:
**Note**: Problem might contain a start code block, if so, you should provide the input parameters strictly in accordance with the signature of the function, like Problem 1. But if there is no start code block, unit tests must not contains any input parameters, your input and output should be a string, like Problem 2.

## Problem 1
You are given a string s and a pattern string p, where p contains exactly two '*' characters.
The '*' in p matches any sequence of zero or more characters.
Return the length of the shortest substring in s that matches p. If there is no such substring, return -1.
Note: The empty substring is considered valid.

Example 1:

Input: s = "abaacbaecebce", p = "ba*c*ce"
Output: 8
Explanation:
The shortest matching substring of p in s is "baecebce".

Example 2:

Input: s = "baccbaadbc", p = "cc*baa*adb"
Output: -1
Explanation:
There is no matching substring in s.

Example 3:

Input: s = "a", p = "**"
Output: 0
Explanation:
The empty substring is the shortest matching substring.

Example 4:

Input: s = "madlogic", p = "*adlogi*"
Output: 6
Explanation:
The shortest matching substring of p in s is "adlogi".


Constraints:

1 <= s.length <= 10^5
2 <= p.length <= 10^5
s contains only lowercase English letters.
p contains only lowercase English letters and exactly two '*'.

Starter Code:
```python
class Solution:
    def shortestMatchingSubstring(self, s: str, p: str) -> int:
```

## Unit Tests
```json
[
    {
        "input": {
            "s": "abaacbaecebce",
            "p": "ba*c*ce"
        },
        "output": 8
    },
    {
        "input": {
            "s": "baccbaadbc",
            "p": "cc*baa*adb"
        },
        "output": -1
    },
    {
        "input": {
            "s": "a",
            "p": "**"
        },
        "output": 0
    },
    {
        "input": {
            "s": "madlogic",
            "p": "*adlogi*"
        },
        "output": 6
    },
    {
        "input": {
            "s": "tbjzlwrnyowcqq",
            "p": "rnyo**w"
        },
        "output": 5
    },
    {
        "input": {
            "s": "otqqkeeycttc",
            "p": "ot*qqkee*ycttc"
        },
        "output": 12
    },
    {
        "input": {
            "s": "wybakrgjn",
            "p": "wyb**jn"
        },
        "output": 9
    }
]
```

## Problem 2
You are given a sequence of M integers A = (A_1, A_2, \\dots, A_M).
Each element of A is an integer between 1 and N, inclusive, and all elements are distinct.
List all integers between 1 and N that do not appear in A in ascending order.

Input

The input is given from Standard Input in the following format:
N M
A_1 A_2 \\dots A_M

Output

Let (X_1, X_2, \\dots, X_C) be the sequence of all integers between 1 and N, inclusive, that do not appear in A, listed in ascending order.
The output should be in the following format:
C
X_1 X_2 \\dots X_C

Constraints


- All input values are integers.
- 1 \\le M \\le N \\le 1000
- 1 \\le A_i \\le N
- The elements of A are distinct.

Sample Input 1

10 3
3 9 2

Sample Output 1

7
1 4 5 6 7 8 10

Here, A=(3,9,2).
The integers between 1 and 10 that do not appear in A, listed in ascending order, are 1,4,5,6,7,8,10.

Sample Input 2

6 6
1 3 5 2 4 6

Sample Output 2

0


No integer between 1 and 6 is missing from A.
In this case, print 0 on the first line and leave the second line empty.

Sample Input 3

9 1
9

Sample Output 3

8
1 2 3 4 5 6 7 8

## Unit Tests
```json
[
    {
        "input": "10 3\n3 9 2",
        "output": "7\n1 4 5 6 7 8 10"
    },
    {
        "input": "6 6\n1 3 5 2 4 6",
        "output": "0"
    },
    {
        "input": "9 1\n9",
        "output": "8\n1 2 3 4 5 6 7 8"
    },
    {
        "input": "115 46\n57 113 66 82 23 51 70 45 10 17 89 13 91 95 54 35 11 6 84 72 58 43 97 92 39 102 77 49 105 27 37 18 34 65 62 47 5 36 79 87 2 101 99 75 81 71",
        "output": "69\n1 3 4 7 8 9 12 14 15 16 19 20 21 22 24 25 26 28 29 30 31 32 33 38 40 41 42 44 46 48 50 52 53 55 56 59 60 61 63 64 67 68 69 73 74 76 78 80 83 85 86 88 90 93 94 96 98 100 103 104 106 107 108 109 110 111 112 114 115"
    },
    {
        "input": "9 3\n8 3 4",
        "output": "6\n1 2 5 6 7 9"
    }
]
```

## Problem 3
You are given a string S consisting of six types of characters: (, ), [, ], <, >.
A string T is called a colorful bracket sequence if it satisfies the following condition:

It is possible to turn T into an empty string by repeating the following operation any number of times (possibly zero):

- If there exists a contiguous substring of T that is one of (), [], or <>, choose one such substring and delete it.
- If the deleted substring was at the beginning or end of T, the remainder becomes the new T.
- Otherwise, concatenate the part before the deleted substring and the part after the deleted substring, and that becomes the new T.


Determine whether S is a colorful bracket sequence.

Input

The input is given from Standard Input in the following format:
S

Output

If S is a colorful bracket sequence, print Yes; otherwise, print No.

Constraints


- S is a string of length between 1 and 2\times 10^5, inclusive.
- S consists of (, ), [, ], <, >.

Sample Input 1

([])<>()

Sample Output 1

Yes

For S=([])<>(), it is possible to turn it into an empty string by repeating the operation as follows:

- Delete the substring [] from the 2nd to the 3rd character in ([])<>(), then concatenate the parts before and after it. The string becomes ()<>().
- Delete the substring () from the 1st to the 2nd character in ()<>(). The string becomes <>().
- Delete the substring <> from the 1st to the 2nd character in <>(). The string becomes ().
- Delete the substring () from the 1st to the 2nd character in (). The string becomes empty.

Thus, S=([])<>() is a colorful bracket sequence, so print Yes.

Sample Input 2

([<)]>

Sample Output 2

No

Since S=([<)]> does not contain (), [], or <> as a contiguous substring, we cannot perform the 1st operation, and in particular S is not a colorful bracket sequence. Therefore, print No.

Sample Input 3

())

Sample Output 3

No

It is impossible to turn S into an empty string by repeating the operations.
Therefore, S is not a colorful bracket sequence, so print No

## Unit Tests
```json
[
    {
        "input": "([])<>()",
        "output": "Yes"
    },
    
    {
        "input": "([<)]>",
        "output": "No"
    },
    {
        "input": "())",
        "output": "No"
    },
    {
        "input": "("*99,
        "output": "No"
    },
    {
        "input": "("*99 + ")"*99,
        "output": "Yes"
    }
]
```


Now please generate the unit tests for the problem.

## Problem
{problem}
"""
