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
from prompts.testing import system_prompt, assistant_prompt
import concurrent.futures
import time
import logging
import re
import json
from tqdm import tqdm
import datetime
import traceback

logger = logging.getLogger(__name__)

MODEL_NAME = "service_dv3_common"
API_KEY = "caa6246b-afbe-4d9b-ab34-87bf9922032b"
BASE_URL = "https://sd265fbi80c6ft26qc5ig.apigateway-cn-beijing.volceapi.com/v1/"

TEMPERATURE = 1.0
MAX_TOKENS = 32768

N = 16
NUM_WORKERS = 64

def create_openai_client(base_url: str, api_key: str):
    http_client = httpx.Client(
        limits=httpx.Limits(
            max_keepalive_connections=20,
            max_connections=40,
            keepalive_expiry=30.0
        ),
        timeout=httpx.Timeout(1800.0)
    )
    
    return openai.OpenAI(
        base_url=base_url,
        api_key=api_key,
        http_client=http_client
    )

client = create_openai_client(BASE_URL, API_KEY)

def make_config(**kwargs):
    return MockArgs(
        output_path='',
        model=MODEL_NAME,
        scenario="codegeneration",
        not_fast=False,
        n=N,
        release_version="release_latest",
        cot_code_execution=False,
        codegen_n=N,
        debug=False,
        continue_existing=False,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        multiprocess=16,
        stop="###",
        continue_existing_with_eval=False,
        use_cache=False,
        cache_batch_size=100,
        evaluate=False,
        num_process_evaluate=1,
        timeout=60,
        openai_timeout=1800,
        start_date="2025-02-01",
        end_date="2025-03-01",
    )
    


def llm_call(prompt: list[dict[str, str]]):

    return client.chat.completions.create(
        model=MODEL_NAME,
        messages=prompt,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

def extract_json_from_text_robust(text: str) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    更健壮的 JSON 提取函数，返回最后一个找到的 JSON 块和错误信息
    
    Args:
        text: 包含 JSON 代码块的文本
        
    Returns:
        tuple: (最后一个提取出的 JSON 对象或 None, 错误信息或 None)
    """
    # 支持多种 JSON 代码块格式
    patterns = [
        r'```json\s*\n(.*?)\n```',  # 标准格式
        r'```json(.*?)```',         # 紧凑格式 ← 这就是您问的这个
        r'`json\s*\n(.*?)\n`',      # 单反引号格式
        r'```\s*json\s*\n(.*?)\n```', # json 前有空格
    ]
    
    all_errors = []
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if matches:
            # 只处理最后一个匹配
            last_match = matches[-1]
            
            try:
                # 清理内容
                cleaned_match = last_match.strip()
                
                # 尝试解析 JSON
                json_obj = json.loads(cleaned_match)
                return json_obj, None
                
            except json.JSONDecodeError as e:
                all_errors.append(f"JSON parsing error: {str(e)}")
                # 如果直接解析失败，尝试修复常见问题
                try:
                    try:
                        # 移除可能的注释和多余空白
                        cleaned = re.sub(r'//.*?\n', '', cleaned_match)
                        cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
                        json_obj = json.loads(cleaned.strip())
                        return json_obj, None
                    except Exception as e:
                        all_errors.append(f"JSON parsing error: {str(e)}")
                        try:
                            json_obj = eval(cleaned_match)
                            return json_obj, None
                        except Exception as e:
                            all_errors.append(f"Eval function parsing error: {str(e)}")
                            continue
                except json.JSONDecodeError as e:
                    all_errors.append(f"Final JSON parsing error: {str(e)}")
                    continue
    
    # 如果没有找到任何JSON块
    if not any(re.search(pattern, text, re.DOTALL | re.IGNORECASE) for pattern in patterns):
        error_msg = "No JSON code blocks found in the response. Please ensure your response contains JSON wrapped in ```json ``` code blocks."
    else:
        error_msg = "JSON parsing failed with the following errors: " + all_errors[-1]
    
    return None, error_msg


def testing_generation_simple(problem: CodeGenerationProblem, idx, max_retry=10, delay=1):
    # if os.path.exists(os.path.join(problem.metadata['file_store_path'], f"{problem.question_title}", "testing.json")):
    #     with open(os.path.join(problem.metadata['file_store_path'], f"{problem.question_title}", "testing.json"), "r") as f:
    #         try:
    #             data = json.load(f)
    #             testing = data["testing_list"][idx]
    #             if testing is not None:
    #                 if (problem.starter_code == "") == (isinstance(testing[0]["input"], str)):
    #                     return data["output_list"][idx], testing
    #         except Exception as e:
    #             pass

    question_content = problem.question_content
    if problem.starter_code != "":
        tmp = f"""
Starter Code:
```python
{problem.starter_code}
```
"""
        question_content = question_content +'\n' + tmp
    
    prompt = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": assistant_prompt.replace("{problem}", question_content)
        }
    ]
    content = None
    testing = None
    for attempt in range(max_retry):
        try:
            res = llm_call(prompt)
            if res.choices[0].finish_reason == "length":
                content = res.choices[0].message.content
                detailed_error_msg = "The input of unit test is too long which can be load by json.loads(), please use a simple python code to generate the input, the python code must can be process by the eval() function in python."
                logger.warning(f"testing_generation_simple 第{attempt+1}次调用失败: {detailed_error_msg}")
                # 将LLM的回复和详细错误提示添加到对话历史中
                prompt.append({
                    "role": "assistant", 
                    "content": content
                })
                prompt.append({
                    "role": "user",
                    "content": detailed_error_msg
                })
            else:
                content = res.choices[0].message.content
                testing, error_msg = extract_json_from_text_robust(content)
                
                if testing is not None:
                    if (problem.starter_code == "") != (isinstance(testing[0]["input"], str)):
                        if problem.starter_code == "":
                            error_msg = "Problem doesn't have starter code, the type of unit test input and output should be a string like Problem 2."
                        else:
                            error_msg = "Problem has starter code, the type of unit test input should be a function parameter dict like Problem 1."
                        logger.warning(f"testing_generation_simple 第{attempt+1}次调用失败: {error_msg}")
                        prompt.append({
                            "role": "assistant", 
                            "content": content
                        })
                        prompt.append({
                            "role": "user",
                            "content": error_msg
                        })
                    else:
                        break
                else:
                    # 使用真正的错误信息作为新的对话内容
                    detailed_error_msg = f"There was an issue with your JSON format. Error details: {error_msg}\n\nPlease fix the JSON format issues and regenerate the unit tests according to the specified format."
                    
                    logger.warning(f"testing_generation_simple 第{attempt+1}次调用失败: {error_msg}")
                    
                    # 将LLM的回复和详细错误提示添加到对话历史中
                    prompt.append({
                        "role": "assistant", 
                        "content": content
                    })
                    prompt.append({
                        "role": "user",
                        "content": detailed_error_msg
                    })
                
        except Exception as e:
            logger.warning(f"testing_generation_simple 第{attempt+1}次调用失败: {e}")
            if attempt < max_retry - 1:
                time.sleep(delay)
            else:
                logger.error(f'Testing Generation Error: {e}')
                raise
                
        if attempt < max_retry - 1:
            time.sleep(delay)
            
    return content, testing

def testing_generation_problem(problem: CodeGenerationProblem):
    # if os.path.exists(os.path.join(problem.metadata['file_store_path'], f"{problem.question_title}", "testing.json")):
    #     with open(os.path.join(problem.metadata['file_store_path'], f"{problem.question_title}", "testing.json"), "r") as f:
    #         data = json.load(f)
    #         return data
    
    results = []
    
    # 提交所有子任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=N) as executor:
        futures = [executor.submit(testing_generation_simple, problem, idx) for idx in range(N)]
        results = [future.result() for future in futures]
    
    good_result = []
    bad_result = []
    for result in results:
        if result[1] is not None:
            good_result.append(result)
        else:
            bad_result.append(result)
    
    testing_result = {
        "question_title": problem.question_title,
        "question_content": problem.question_content,
        "output_list": [result[0] for result in good_result+bad_result],
        "testing_list": [result[1] for result in good_result+bad_result]
    }
    os.makedirs(os.path.join(problem.metadata['file_store_path'], f"{problem.question_title}"), exist_ok=True)
    with open(os.path.join(problem.metadata['file_store_path'], f"{problem.question_title}", "testing.json"), "w") as f:
        json.dump(testing_result, f, indent=4)
    return testing_result
        
def testing_generation(dataset: list[CodeGenerationProblem]):
    """生成测试用例，带进度条"""
    print("🧪 开始生成测试用例...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # 提交所有任务
        futures = {executor.submit(testing_generation_problem, problem): problem 
                  for problem in dataset}
        
        # 使用tqdm显示进度
        results = []
        with tqdm(total=len(futures), desc="📝 生成测试", 
                 bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}',
                 colour='green') as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    pbar.set_postfix({"完成": f"{len(results)}/{len(dataset)}"})
                    pbar.update(1)
                except Exception as e:
                    problem = futures[future]
                    logger.error(f"测试生成失败 {problem.question_title}: {e}")
                    pbar.update(1)
    
    print(f"✅ 测试用例生成完成: {len(results)}/{len(dataset)}")
    return results

def solution_generation_problem(problem: CodeGenerationProblem):
    mock_args = make_config()
    os.makedirs(os.path.join(problem.metadata['file_store_path'], f"{problem.question_title}"), exist_ok=True)
    mock_args.output_path = os.path.join(problem.metadata['file_store_path'], f"{problem.question_title}", "solution.json")
    generate(mock_args, problem)

def solution_generation(dataset: list[CodeGenerationProblem]):
    """生成解决方案，带进度条"""
    print("💡 开始生成解决方案...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # 提交所有任务
        futures = {executor.submit(solution_generation_problem, problem): problem 
                  for problem in dataset}
        
        # 使用tqdm显示进度
        results = []
        with tqdm(total=len(futures), desc="🔧 生成方案", 
                 bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}',
                 colour='blue') as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    pbar.set_postfix({"完成": f"{len(results)}/{len(dataset)}"})
                    pbar.update(1)
                except Exception as e:
                    problem = futures[future]
                    logger.error(f"方案生成失败 {problem.question_title}: {e}")
                    pbar.update(1)
    
    print(f"✅ 解决方案生成完成: {len(results)} 个问题")
    return results

def verify(problem, solution_id, testing_id):
    solutions = []
    testings = []
    try:
        with open(os.path.join(problem.metadata['file_store_path'], f"{problem.question_title}", "solution.json"), "r") as f:
            data = json.load(f)
            solutions = data["code_list"]
        if testing_id != "ground_truth":
            with open(os.path.join(problem.metadata['file_store_path'], f"{problem.question_title}", "testing.json"), "r") as f:
                data = json.load(f)
                testings = data["testing_list"]
                testing = testings[int(testing_id)]
                inputs = []
                outputs = []  # 修复缩进问题
            if testing is None:
                return {
                    "question_title": problem.question_title,
                    "solution_id": solution_id,
                    "testing_id": testing_id,
                    "result": False,
                    "unknown_error": True,
                    "metadata": {}
                }
            for test in testing:
                inputs.append(test["input"])
                outputs.append(test["output"])
            
            in_out = {"inputs": inputs, "outputs": outputs, "fn_name": problem.metadata.get('func_name', None)}
            sample = {"input_output": json.dumps(in_out)}
        else:
            sample = problem.get_evaluation_sample()

        solution = solutions[int(solution_id)]
    except Exception as e:
        logger.error(f"Verify Error: {e}")
        logger.error(traceback.format_exc())
        return {
                "question_title": problem.question_title,
                "solution_id": solution_id,
                "testing_id": testing_id,
                "result": False,
                "unknown_error": True,
                "metadata": {}
            }
    try:
        res, metadata = run_test(sample, solution, debug=False, timeout=60)
        result = True if metadata.get("execution time", None) != None else False
        unknown_error = False
    except Exception as e:
        logger.error(f"Verify Error: {e}")
        logger.error(traceback.format_exc())
        result = False
        unknown_error = True
        metadata = {}
    
    output = {
        "question_title": problem.question_title,
        "solution_id": solution_id,
        "testing_id": testing_id,
        "result": result,
        "unknown_error": unknown_error,
        "metadata": metadata
    }

    return output
    
def final_check_problem(problem: CodeGenerationProblem, sort_solution_id_list: list[int], sort_testing_id_list: list[int], top_s = 8):
    correct_solution_id_list = sort_solution_id_list[:top_s]
    wrong_solution_id_list = sort_solution_id_list[-top_s:]
    # correct_solution_id = sort_solution_id_list[0]
    # wrong_solution_id = sort_solution_id_list[-1]
    correct_testing_id = sort_testing_id_list[0]
    
    correct_output_list = []
    wrong_output_list = []
    for correct_solution_id in correct_solution_id_list:
        correct_output = verify(problem, correct_solution_id, correct_testing_id)
        correct_output_list.append(correct_output)
    for wrong_solution_id in wrong_solution_id_list:
        wrong_output = verify(problem, wrong_solution_id, correct_testing_id)
        wrong_output_list.append(wrong_output)
        
    ground_correct_output_list = []
    ground_wrong_output_list = []
    for correct_solution_id in correct_solution_id_list:
        ground_correct_output = verify(problem, correct_solution_id, "ground_truth")
        ground_correct_output_list.append(ground_correct_output)
    for wrong_solution_id in wrong_solution_id_list:
        ground_wrong_output = verify(problem, wrong_solution_id, "ground_truth")
        ground_wrong_output_list.append(ground_wrong_output)
        
    correct_result_list = [output['result'] for output in correct_output_list]
    wrong_result_list = [output['result'] for output in wrong_output_list]
    ground_correct_result_list = [output['result'] for output in ground_correct_output_list]
    ground_wrong_result_list = [output['result'] for output in ground_wrong_output_list]
    
    # correct_output = verify(problem, correct_solution_id, correct_testing_id)
    # wrong_output = verify(problem, wrong_solution_id, correct_testing_id)
    
    # ground_correct_output = verify(problem, correct_solution_id, "ground_truth")
    # ground_wrong_output = verify(problem, wrong_solution_id, "ground_truth")
    
    # correct_result = correct_output['result']
    # wrong_result = wrong_output['result']
    # ground_correct_result = ground_correct_output['result']
    # ground_wrong_result = ground_wrong_output['result']
    
    # unknown_error = correct_output_list[0]['unknown_error'] or wrong_output_list[-1]['unknown_error'] or ground_correct_output_list[0]['unknown_error'] or ground_wrong_output_list[-1]['unknown_error']
    unknown_error = False
    output = {
        "question_title": problem.question_title,
        "correct_solution_id": correct_solution_id,
        "correct_testing_id": correct_testing_id,
        "wrong_solution_id": wrong_solution_id,
        "correct_result": correct_result_list,
        "wrong_result": wrong_result_list,
        "ground_correct_result": ground_correct_result_list[0],
        "ground_wrong_result": ground_wrong_result_list[-1],
        "unknown_error": unknown_error,
        # "result": ground_correct_result and correct_result and (ground_wrong_result == wrong_result)
        "result": ground_correct_result_list[0] and (ground_correct_result_list == correct_result_list) and (ground_wrong_result_list == wrong_result_list)
    }
    os.makedirs(os.path.join(problem.metadata['file_store_path'], f"{problem.question_title}"), exist_ok=True)
    with open(os.path.join(problem.metadata['file_store_path'], f"{problem.question_title}", "final_check.json"), "w") as f:
        json.dump(output, f, indent=4)
    return output

def final_check(tasks):
    """最终检查，带进度条"""
    print("🔍 开始最终检查...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # 提交所有任务
        futures = {executor.submit(final_check_problem, task[0], task[1], task[2]): task 
                  for task in tasks}
        
        # 使用tqdm显示进度
        results = []
        with tqdm(total=len(futures), desc="✅ 最终检查", 
                 bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}',
                 colour='yellow') as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    # 实时显示通过率
                    passed = sum(1 for r in results if r['ground_correct_result'])
                    pass_rate = passed / len(results) * 100
                    pbar.set_postfix({
                        "完成": f"{len(results)}/{len(tasks)}",
                        "通过率": f"{pass_rate:.1f}%"
                    })
                    pbar.update(1)
                except Exception as e:
                    task = futures[future]
                    logger.error(f"最终检查失败 {task[0].question_title}: {e}")
                    pbar.update(1)
    score = []
    answer_score = []
    for result in results:
        if not result['unknown_error']:
            if result['ground_correct_result']:
                answer_score.append(1)
            else:
                answer_score.append(0)
            if result['result']:
                score.append(1)
            else:
                score.append(0)
    final_score = sum(score) / len(score)
    final_answer_score = sum(answer_score) / len(answer_score)
    
    print(f"🎯 最终评分: {final_score:.3f} ({score}/{len(score)})")
    print(f"🎯 答案评分: {final_answer_score:.3f} ({answer_score}/{len(answer_score)})")
    return final_score, final_answer_score

def generation_final_check_problem(problem: CodeGenerationProblem, sort_solution_id_list: list[int], sort_testing_id_list: list[int]):
    correct_solution_id = sort_solution_id_list[0]
    correct_testing_id = sort_testing_id_list[0]
    
    correct_output = verify(problem, correct_solution_id, correct_testing_id)
    
    correct_result = correct_output['result']
    
    unknown_error = correct_output['unknown_error']
    output = {
        "question_title": problem.question_title,
        "correct_solution_id": correct_solution_id,
        "correct_testing_id": correct_testing_id,
        "correct_result": correct_result,
        "unknown_error": unknown_error,
        "result": correct_result
    }
    os.makedirs(os.path.join(problem.metadata['file_store_path'], f"{problem.question_title}"), exist_ok=True)
    with open(os.path.join(problem.metadata['file_store_path'], f"{problem.question_title}", "generation_final_check.json"), "w") as f:
        json.dump(output, f, indent=4)
    return output

def generation_final_check(tasks: list[tuple[CodeGenerationProblem, list[int], list[int]]]):
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # 提交所有任务
        futures = {executor.submit(generation_final_check_problem, task[0], task[1], task[2]): task 
                  for task in tasks}
        
        # 使用tqdm显示进度
        results = []
        with tqdm(total=len(futures), desc="✅ 生成最终检查", 
                 bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}',
                 colour='yellow') as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    # 实时显示通过率
                    passed = sum(1 for r in results if r['result'])
                    pass_rate = passed / len(results) * 100
                    pbar.set_postfix({
                        "完成": f"{len(results)}/{len(tasks)}",
                        "通过率": f"{pass_rate:.1f}%"
                    })
                    pbar.update(1)
                except Exception as e:
                    task = futures[future]
                    logger.error(f"最终检查失败 {task[0].question_title}: {e}")
                    pbar.update(1)
    score = []
    for result in results:
        if not result['unknown_error']:
            if result['result']:
                score.append(1)
            else:
                score.append(0)
    final_score = sum(score) / len(score)
    
    print(f"🎯 最终评分: {final_score:.3f} ({score}/{len(score)})")
    return final_score

def strategy_wrapper(program_path, instance, solution_id_list, testing_id_list):
    """包装函数，用于在子进程中动态导入并执行strategy函数"""
    import importlib.util
    
    # 在子进程中重新导入模块
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    
    # 在子进程中也需要重新定义verify函数，因为它也不能被pickle
    # 这里我们直接调用strategy函数，让它处理verify的逻辑
    return program.strategy(instance, solution_id_list, testing_id_list, verify)

def evaluate(program_path: str):
    """主评估函数，带完整进度显示"""
    print("🚀 开始评估流程...")
    
    current_time = datetime.datetime.now()
    FILE_STORE = "openevolve/livecode_file_store/init"
    
    print(f"📁 文件存储路径: {FILE_STORE}")
    
    # 构建基准测试
    print("📋 构建基准测试...")
    args = make_config()
    benchmark, _ = build_prompt_benchmark(args)
    for instance in benchmark:
        instance.metadata['file_store_path'] = FILE_STORE
    
    print(f"📊 总共 {len(benchmark)} 个测试问题")
    
    # 生成测试用例
    # testing_generation(benchmark)
    
    # 生成解决方案
    # solution_generation(benchmark)
    
    # 验证阶段
    print("🔄 开始验证阶段...")
    final_check_datasets = []
    
    with tqdm(benchmark, desc='🔍 策略验证', 
             bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}',
             colour='cyan') as pbar:
        for instance in pbar:
            if os.path.exists(os.path.join(FILE_STORE, f"{instance.question_title}", "sort_list.json")):
                with open(os.path.join(FILE_STORE, f"{instance.question_title}", "sort_list.json"), "r") as f:
                    data = json.load(f)
                    sort_solution_id_list = data["sort_solution_id_list"]
                    sort_testing_id_list = data["sort_testing_id_list"]
            else:
                solution_id_list = testing_id_list = [str(i) for i in range(N)]
                sort_solution_id_list, sort_testing_id_list = strategy_wrapper(
                    program_path, instance, solution_id_list, testing_id_list
                )
                with open(os.path.join(FILE_STORE, f"{instance.question_title}", "sort_list.json"), "w") as f:
                    json.dump({"sort_solution_id_list": sort_solution_id_list, "sort_testing_id_list": sort_testing_id_list}, f, indent=4)
            final_check_datasets.append((instance, sort_solution_id_list, sort_testing_id_list))
            pbar.set_postfix({"问题": instance.question_title[:20] + "..."})
    
    # 最终检查
    final_score, answer_score = final_check(final_check_datasets)
    
    print("🎉 评估完成!")
    print(f"📈 最终得分: {final_score:.3f}")
    print(f"🎯 答案评分: {answer_score:.3f}")
    
    return {"score": final_score}

def generation(dataset_path : str, strategy_path : str):
    current_time = datetime.datetime.now()
    FILE_STORE_PATH = 'livecode_data/file_store' + f'/{current_time.strftime("%Y%m%d%H%M%S")}'
    # args = make_config()
    new_datasets = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            instance = json.loads(line)
            livecode_instance = CodeGenerationProblem(
                question_title=instance['question_title'],
                question_content=instance['question_content'],
                starter_code=instance.get('starter_code', ''),
                metadata=json.dumps(instance['metadata']),
                public_test_cases='[]',
                private_test_cases='[]',
                difficulty=instance['difficulty'],
                platform=instance.get('platform', 'leetcode'),
                contest_id=instance.get('contest_id', 'fake_contest_id'),
                contest_date=instance.get('contest_date', '2030-01-01'),
                question_id=instance.get('idx', '')
            )
            livecode_instance.metadata['file_store_path'] = FILE_STORE_PATH
            new_datasets.append(livecode_instance)
    benchmark = new_datasets
    print(f"📊 总共 {len(benchmark)} 个测试问题")
    
    # 生成测试用例
    testing_generation(benchmark)
    
    # 生成解决方案
    solution_generation(benchmark)
    
     # 验证阶段
    print("🔄 开始验证阶段...")
    final_check_datasets = []
    
    with tqdm(benchmark, desc='🔍 策略验证', 
             bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}',
             colour='cyan') as pbar:
        for instance in pbar:
            if os.path.exists(os.path.join(FILE_STORE_PATH, f"{instance.question_title}", "sort_list.json")):
                with open(os.path.join(FILE_STORE_PATH, f"{instance.question_title}", "sort_list.json"), "r") as f:
                    data = json.load(f)
                    sort_solution_id_list = data["sort_solution_id_list"]
                    sort_testing_id_list = data["sort_testing_id_list"]
            else:
                solution_id_list = testing_id_list = [str(i) for i in range(N)]
                sort_solution_id_list, sort_testing_id_list = strategy_wrapper(
                    strategy_path, instance, solution_id_list, testing_id_list
                )
            with open(os.path.join(FILE_STORE_PATH, f"{instance.question_title}", "sort_list.json"), "w") as f:
                json.dump({"sort_solution_id_list": sort_solution_id_list, "sort_testing_id_list": sort_testing_id_list}, f, indent=4)
            final_check_datasets.append((instance, sort_solution_id_list, sort_testing_id_list))
            pbar.set_postfix({"问题": instance.question_title[:20] + "..."})
    
    final_score = generation_final_check(final_check_datasets)
    
    print("🎉 评估完成!")
    print(f"📈 最终得分: {final_score:.3f}")
    
    return {"score": final_score}
    
if __name__ == "__main__":
    evaluate("livecode/init_program.py")
