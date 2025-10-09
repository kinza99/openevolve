import openai
import re
import json
import concurrent.futures
from multiprocessing import Manager
import datetime
import importlib.util
from tqdm import tqdm
import warnings
import random
import logging
import time
import socket
import subprocess 
import httpx
import litellm
import traceback
from collections import defaultdict
import pandas as pd
from datasets import load_dataset
from evaluator import type_check, llm_call, get_llm_config, MODEL_NAME
from prompts.question import system_prompt, user_prompt
import logging
import re
import random
logger = logging.getLogger(__name__)
NUM_REPEAT = 16
llm_config = get_llm_config(MODEL_NAME)

def parse_template(text):
    """
    解析模板文本，提取各个部分
    
    Args:
        text (str): 包含模板的文本
    
    Returns:
        dict: 包含解析结果的字典
    """
    # 使用re.DOTALL标志让.匹配换行符
    pattern = r"### Query:\s*\n(.*?)\n\n### Init:\s*\n(.*?)\n\n### Type:\s*\n(.*?)\n\n### Comparison_method:\s*\n(.*?)(?=\n\n|\Z)"
    
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        return {
            'description': match.group(1).strip(),
            'init': match.group(2).strip(),
            'type': match.group(3).strip(),
            'comparison_method': match.group(4).strip()
        }
    else:
        return None

def get_prompt(instance : pd.Series):
    return [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt.format(example_query=instance.description, example_init=instance.init, example_type=instance.type, example_comparison_method=instance.comparison_method)}]

def llm_call_with_retry(prompt : list[dict[str, str]], max_retry : int = 10, delay : float = 2.0):
    for attempt in range(max_retry):
        try:
            res = llm_call(prompt)
            content = res['choices'][0]['message']['content']
            new_instance = parse_template(content)
            if new_instance is None:
                logger.error(f'Failed to parse model answer: {res}')
                return None
            return new_instance
        except Exception as e:
            logger.warning(f"llm_call 第{attempt+1}次调用失败: {e}")
            if attempt < max_retry - 1:
                time.sleep(random.uniform(0, 5))
            else:
                logger.error(f"llm_call 第{attempt+1}次调用失败: {e}")
                return None
    return None

def generate_question():
    FILE_STORE_PATH = 'verifier_data/questions'
    dataset = load_dataset('iFurySt/AgentBench')
    agent_bench_tests = dataset['osbench'].to_pandas()
    new_datasets = []
    for _, instance in agent_bench_tests.iterrows():
        instance = instance.copy()
        instance['type'] = None
        instance['file_store_path'] = FILE_STORE_PATH
        instance['idx'] = len(new_datasets)
        new_datasets.append(instance)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(new_datasets)) as executor:
        futures = [executor.submit(type_check, instance) for instance in new_datasets]
        for future in tqdm(concurrent.futures.as_completed(futures), desc='Type Check', total=len(futures), dynamic_ncols=True):
            future.result()

    def instance_check(instance : pd.Series):
        if instance.type == 'QA' and instance.get_agent_result is None:
            return True
        elif instance.type == 'EXEC' and instance.get_agent_result is not None:
            return True
        else:
            instance.type = 'WRONG'
            return False

    valid_instances = [instance for instance in new_datasets if instance_check(instance)]
    
    prompts = []
    for instance in valid_instances:
        prompt = get_prompt(instance)
        prompts.extend([prompt]*NUM_REPEAT)
    
    p0 = prompts[0]
    p0_instance = llm_call_with_retry(p0)
    print(p0_instance)
    
    new_instances = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
        futures = [executor.submit(llm_call_with_retry, prompt) for prompt in prompts]
        for future in tqdm(concurrent.futures.as_completed(futures), desc='LLM Call', total=len(futures), dynamic_ncols=True):
            new_instance = future.result()
            if new_instance is not None:
                tmp = {
                    "instance_id": f'instance_{len(new_instances)}',
                    "description": new_instance['description'],
                    "init": new_instance['init'],
                    "type": new_instance['type'],
                    "comparison_method": new_instance['comparison_method'],
                    "idx": len(new_instances)
                }
                new_instances.append(tmp)
            else:
                logger.warning(f'Failed to generate new instance: {future.result()}')

    with open(FILE_STORE_PATH+'/new_dataset'+datetime.datetime.now().strftime('%Y%m%d%H%M%S')+'.jsonl', 'w', encoding='utf-8') as f:
        for instance in new_instances:
            f.write(json.dumps(instance, ensure_ascii=False)+'\n')
        
    
    
if __name__ == '__main__':
    generate_question()