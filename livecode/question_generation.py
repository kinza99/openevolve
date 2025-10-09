import os
import sys
# 将当前文件所在目录添加到 Python 模块搜索路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from lcb_runner.runner.main import MockArgs, generate
from litellm import completion
from typing import Any, Optional, Dict
import httpx
import openai
from lcb_runner.benchmarks import CodeGenerationProblem
from lcb_runner.runner.scenario_router import build_prompt_benchmark
from lcb_runner.evaluation.testing_util import run_test
from lcb_runner.lm_styles import LMStyle
from lcb_runner.utils.extraction_utils import extract_code
from evaluator import make_config, llm_call
from prompts.question import system_prompt, func_prompt, std_prompt
import concurrent.futures
import time
import logging
import re
import json
from tqdm import tqdm
import datetime
import traceback
import random

logger = logging.getLogger(__name__)
NUM_REPEAT=10

def get_prompt(instance):
    if instance.starter_code == "":
        return "std", [{
            "role": "system",
            "content": system_prompt
        },{
            "role": "user",
            "content": std_prompt.format(example_description=instance.question_content, example_title=instance.question_title)
        }]
    else:
        return "func", [{
            "role": "system",
            "content": system_prompt
        },{
            "role": "user",
            "content": func_prompt.format(example_description=instance.question_content, example_title=instance.question_title, example_starter_code=instance.starter_code, example_function_name=instance.metadata['func_name'])
        }]
def parse_template(the_type, text):
    """
    解析模板文本，提取各个部分
    
    Args:
        text (str): 包含模板的文本
    
    Returns:
        dict: 包含解析结果的字典
    """
    if the_type == "std":
        # 使用re.DOTALL标志让.匹配换行符
        pattern = r"### Description\s*\n(.*?)\n\n### Title\s*\n(.*?)(?=\n\n|\Z)"
    else:
        # 使用re.DOTALL标志让.匹配换行符
        pattern = r"### Description\s*\n(.*?)\n\n### Title\s*\n(.*?)\n\n### Starter_Code\s*\n(.*?)\n\n### Function_name\s*\n(.*?)(?=\n\n|\Z)"
    
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        if the_type == "std":
            return {
                'description': match.group(1).strip(),
                'title': match.group(2).strip(),
            }
        else:
            starter_code = extract_code(match.group(3).strip(), LMStyle.DeepSeekAPI)
            # 检查starter_code是否以正确的缩进结尾
            if starter_code and not starter_code.endswith('\n        '):
                starter_code = starter_code.strip() + '\n        '
            if match.group(4).strip() not in starter_code:
                return None
            return {
                'description': match.group(1).strip(),
                'title': match.group(2).strip(),
                'starter_code': starter_code,
                'function_name': match.group(4).strip(),
            }
    else:
        return None

def llm_call_with_retry(the_type, prompt : list[dict[str, str]], max_retry : int = 10, delay : float = 2.0):
    for attempt in range(max_retry):
        try:
            res = llm_call(prompt)
            content = res.choices[0].message.content
            new_instance = parse_template(the_type, content)
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
    OUTPUT_PATH = "livecode_data/questions"
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    args = make_config()
    args.start_date="2025-02-01"
    args.end_date="2025-05-01"
    benchmark, _ = build_prompt_benchmark(args)
    
    # 为每个原始实例生成题目
    new_instances = []
    global_seen_titles = set()  # 全局去重
    total_expected = len(benchmark) * NUM_REPEAT  # 期望生成的总题目数
    
    # 全局进度条
    global_pbar = tqdm(total=total_expected, desc='总体进度', position=0, dynamic_ncols=True)
    
    for i, instance in enumerate(benchmark):
        logger.info(f"处理第 {i+1}/{len(benchmark)} 个原始实例")
        prompt_type, prompt_messages = get_prompt(instance)
        
        # 为当前实例生成指定数量的题目
        instance_questions = []
        attempt_count = 0
        max_total_attempts = NUM_REPEAT * 10  # 最大总尝试次数
        
        # 为当前实例创建一个持久的进度条
        instance_pbar = tqdm(total=NUM_REPEAT, 
                           desc=f'实例{i+1}/{len(benchmark)}', 
                           position=1, 
                           dynamic_ncols=True,
                           leave=False)
        
        while len(instance_questions) < NUM_REPEAT and attempt_count < max_total_attempts:
            # 计算还需要生成多少个题目
            remaining_needed = NUM_REPEAT - len(instance_questions)
            # 批量生成，考虑到可能的重复，生成更多一些
            batch_size = 128
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
                futures = [executor.submit(llm_call_with_retry, prompt_type, prompt_messages) for _ in range(batch_size)]
                
                for future in concurrent.futures.as_completed(futures):
                    if len(instance_questions) >= NUM_REPEAT:
                        # 已经生成足够数量的题目，取消剩余任务
                        for remaining_future in futures:
                            if not remaining_future.done():
                                remaining_future.cancel()
                        break
                        
                    new_instance = future.result()
                    attempt_count += 1
                    
                    if new_instance is not None:
                        title = new_instance['title']
                        if title not in global_seen_titles:
                            global_seen_titles.add(title)
                            if new_instance.get('starter_code') is None:
                                metadata = {}
                            else:
                                metadata = {"func_name": new_instance.get('function_name')}
                            tmp = {
                                "question_content": new_instance['description'],
                                "question_title": new_instance['title'],
                                "starter_code": new_instance.get('starter_code', ''),
                                "metadata": metadata,
                                "idx": len(new_instances) + len(instance_questions),
                                "difficulty": 'medium',
                                "source_instance_idx": i
                            }
                            instance_questions.append(tmp)
                            # 更新全局进度
                            global_pbar.update(1)
                            # 更新实例进度
                            instance_pbar.update(1)
                        else:
                            logger.debug(f'跳过重复标题: {title}')
                    else:
                        logger.warning(f'生成实例失败')
        
        # 关闭当前实例的进度条
        instance_pbar.close()
        
        # 将当前实例生成的题目添加到总列表
        new_instances.extend(instance_questions)
        
        if len(instance_questions) == NUM_REPEAT:
            logger.info(f"实例 {i+1} 成功生成 {len(instance_questions)}/{NUM_REPEAT} 个唯一题目")
        else:
            logger.warning(f"实例 {i+1} 只生成了 {len(instance_questions)}/{NUM_REPEAT} 个唯一题目 (尝试了{attempt_count}次)")
            # 如果没有达到目标，也要更新全局进度条到应该达到的位置
            missing = NUM_REPEAT - len(instance_questions)
            global_pbar.update(missing)

    global_pbar.close()

    # 重新设置idx
    for i, instance in enumerate(new_instances):
        instance['idx'] = i

    logger.info(f"总共生成了 {len(new_instances)} 个题目，来自 {len(benchmark)} 个原始实例")
    
    with open(OUTPUT_PATH+'/new_dataset'+datetime.datetime.now().strftime('%Y%m%d%H%M%S')+'.jsonl', 'w', encoding='utf-8') as f:
        for instance in new_instances:
            f.write(json.dumps(instance, ensure_ascii=False)+'\n')



if __name__ == "__main__":
    generate_question()