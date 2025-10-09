from pydantic import BaseModel
from openhands.core.config import LLMConfig, SandboxConfig
import asyncio
import os
import re
import tempfile
from typing import Any

import pandas as pd
from datasets import load_dataset
import litellm
from litellm import completion
import subprocess
import threading
import sys
import os
import docker
# 将当前文件所在目录添加到 Python 模块搜索路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from helper import (
    FAKE_RESPONSES,
    INST_SUFFIXES,
    compare_results,
    create_sh_file,
)
from shared import (
    EvalMetadata,
    EvalOutput,
    compatibility_for_eval_history_pairs,
    get_default_sandbox_config_for_eval,
    make_metadata,
    prepare_dataset,
    reset_logger_for_multiprocessing,
    run_evaluation,
    delete_container,
    delete_all_stopped_containers,
    get_runtime_id,
    update_llm_config_for_completions_logging
)
from openhands.controller.state.state import State
from openhands.core.config import (
    AppConfig,
    get_llm_config_arg,
    parse_arguments,
)
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime, run_controller
from openhands.events.action import AgentFinishAction, CmdRunAction, MessageAction, FileEditAction
from openhands.events.serialization.event import event_from_dict
from openhands.events.observation import CmdOutputObservation
from openhands.runtime.base import Runtime
from openhands.utils.async_utils import call_async_from_sync
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

os.environ["LITELLM_LOG"] = "ERROR"

logger.setLevel(logging.ERROR)
logging.getLogger('litellm').setLevel(logging.ERROR)
logging.getLogger('litellm.llms').setLevel(logging.ERROR)
logging.getLogger('litellm.utils').setLevel(logging.ERROR)
logging.getLogger('litellm.router').setLevel(logging.ERROR)
logging.getLogger('litellm.proxy').setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
# 忽略所有 UserWarning 警告
warnings.filterwarnings("ignore", category=UserWarning)

OH_DEFAULT_AGENT = 'CodeActAgent'
OH_MAX_ITERATIONS = 50

NUM_WORKERS = 16
N=1
MODEL_NAME = 'qwen3-8b-base-1e5-random-nothink'

# 添加新的配置参数
BATCH_SIZE = 32  # 每批处理的任务数量
BATCH_DELAY = 5  # 批次间延迟时间（秒）

def get_occupied_ports():
    """
    获取当前系统所有被占用的端口号
    
    Returns:
        set: 被占用的端口号集合
    """
    occupied_ports = set()
    
    try:
        # 使用ss命令获取所有占用的端口（更现代化的方式）
        result = subprocess.run(['lsof', '-i', '-P', '-n'], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')[1:]  # 跳过标题行
        
        for line in lines:
            parts = line.split()
            if len(parts) >= 9:
                # 解析本地地址，格式可能是 *:port 或 ip:port 或 [::]:port
                local_addr = parts[-2]
                ip_list = local_addr.split('->')
                for ip in ip_list:
                    port_str = ip.split(':')[-1]
                    try:
                        port = int(port_str)
                        occupied_ports.add(port)
                    except ValueError:
                        continue
    except Exception as e:
        logger.error(f'Error getting occupied ports: {e}')
        occupied_ports = set()
    
    return list(occupied_ports)


def get_llm_config(model_name: str):
    if model_name == 'deepseek-r1':
        model_name = "hosted_vllm/service_r10528_forbowen"
        base_url = "https://sd1egm6r54gpj4to1urkg.apigateway-cn-beijing-inner.volceapi.com/v1"
        api_key = "caa6246b-afbe-4d9b-ab34-87bf9922032b"
        proxy = None
    elif model_name == 'deepseek-v3':
        model_name = "openai/deepseek"
        base_url = "https://180.163.156.43:21020/dsv3/v1/"
        api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoibGl5aW5pbmciLCJleHAiOjE3OTg3NjE2MDB9.t90tauUt9YrZzYg9yi1_0GFBDBO9zYzauNKKxmwCto8"
        proxy = 'pjlab'
    elif model_name in ['qwen3-8b', 'qwen3-4b', 'llama3-1-8b']:
        model_name = f"hosted_vllm/{model_name}"
        base_url = "http://localhost:8000/v1"
        api_key = "caa6246b-afbe-4d9b-ab34-87bf9922032b"
        proxy = None
    else:
        model_name = f"hosted_vllm/{model_name}"
        base_url = "http://47.100.57.163:40088/v1"
        api_key = "caa6246b-afbe-4d9b-ab34-87bf9922032b"
        proxy = None
    llm_config = LLMConfig(
        model=model_name,
        temperature=0.0,
        log_completions=True,
        base_url=base_url,
        api_key=api_key,
        proxy=proxy,
        timeout=300,
        retry_max_wait=300,
        retry_min_wait=5,
        num_retries=10,
        retry_multiplier=1.5,
    )
    return llm_config

def get_config(
    metadata: EvalMetadata,
) -> AppConfig:
    sandbox_config = get_default_sandbox_config_for_eval()
    sandbox_config.base_container_image = 'python:3.12-slim'
    sandbox_config.runtime_container_image = 'docker.all-hands.dev/all-hands-ai/runtime:0.28-nikolaik'
    config = AppConfig(
        default_agent=metadata.agent_class,
        run_as_openhands=False,
        runtime=os.environ.get('RUNTIME', 'docker'),
        max_iterations=metadata.max_iterations,
        sandbox=sandbox_config,
        # do not mount workspace
        workspace_base=None,
        workspace_mount_path=None,
    )
    config.set_llm_config(metadata.llm_config)
    agent_config = config.get_agent_config(metadata.agent_class)
    agent_config.enable_prompt_extensions = False
    return config

def get_metadata(
    llm_config: LLMConfig,
    dataset_name: str,
    agent_class: str,
    max_iterations: int,
):
    return EvalMetadata(
        llm_config=llm_config,
        dataset_name=dataset_name,
        agent_class=agent_class,
        max_iterations=max_iterations,
    )
    
metadata = get_metadata(
    llm_config=get_llm_config(MODEL_NAME),
    dataset_name='AgentBench-OS',
    agent_class=OH_DEFAULT_AGENT,
    max_iterations=OH_MAX_ITERATIONS,
)

# 创建带连接池限制的OpenAI客户端
def create_openai_client(base_url: str, api_key: str):
    http_client = httpx.Client(
        limits=httpx.Limits(
            max_keepalive_connections=20,
            max_connections=40,
            keepalive_expiry=30.0
        ),
        timeout=httpx.Timeout(60.0)
    )
    
    return openai.OpenAI(
        base_url=base_url,
        api_key=api_key,
        http_client=http_client
    )

client = create_openai_client(metadata.llm_config.base_url, metadata.llm_config.api_key.get_secret_value())

def run_action_with_retry(runtime: Runtime, action: CmdRunAction, instance: pd.Series, max_retries: int = 30, delay: float = 10.0):
    """
    执行动作直到成功或达到最大重试次数
    
    Args:
        runtime: Runtime 实例
        action: 要执行的动作
        max_retries: 最大重试次数
        delay: 重试间隔（秒）
    
    Returns:
        CmdOutputObservation: 成功执行后的观察结果
    
    Raises:
        RuntimeError: 如果达到最大重试次数仍然失败
    """
    for attempt in range(max_retries + 1):
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        
        if obs.exit_code == 0:
            return obs
        if attempt < max_retries:
            logger.warning(f'Instance {instance.instance_id} Command failed with exit code {obs.exit_code}, Content: {obs.content}, retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries + 1})')
            time.sleep(delay)
        else:
            logger.error(f'Instance {instance.instance_id} Command failed after {max_retries + 1} attempts with exit code {obs.exit_code}, Content: {obs.content}')
            logger.error(traceback.format_exc())
            raise RuntimeError(f'Command failed after {max_retries + 1} attempts with exit code {obs.exit_code}')
    
def copy_file_with_retry(instance: pd.Series, runtime: Runtime, host_src: str, sandbox_dest: str, max_retries: int = 5, delay: float = 5):
    """
    带重试机制的文件copy操作，确保文件成功传输到容器中
    
    Args:
        runtime: Runtime 实例
        host_src: 源文件路径
        sandbox_dest: 目标路径
        instance: 实例信息
        max_retries: 最大重试次数
        delay: 重试间隔（秒）
    
    Returns:
        bool: 是否copy成功
    
    Raises:
        RuntimeError: 如果达到最大重试次数仍然失败
    """
    for attempt in range(max_retries + 1):
        try:
            # 确保源文件存在
            if not os.path.exists(host_src):
                raise FileNotFoundError(f'Source file {host_src} does not exist')
            
            logger.info(f'Instance {instance.instance_id}: Attempting to copy {host_src} to {sandbox_dest} (attempt {attempt + 1}/{max_retries + 1})')
            
            # 执行copy操作
            runtime.copy_to(host_src, sandbox_dest)
            
            # 验证文件是否成功copy - 获取文件名
            filename = os.path.basename(host_src)
            sandbox_file_path = os.path.join(sandbox_dest, filename).replace('\\', '/')
            
            # 验证文件是否存在
            check_action = CmdRunAction(command=f'test -f "{sandbox_file_path}" && echo "FILE_EXISTS" || echo "FILE_NOT_EXISTS"')
            obs = run_action_with_retry(runtime, check_action, instance)
            
            if obs.exit_code == 0 and "FILE_EXISTS" in obs.content:
                logger.info(f'Instance {instance.instance_id}: Successfully copied {host_src} to {sandbox_dest}')
                return True
            else:
                logger.warning(f'Instance {instance.instance_id}: File verification failed - file not found in container: {obs.content}')
                if attempt < max_retries:
                    time.sleep(delay)
                    continue
                    
        except Exception as e:
            logger.warning(f'Instance {instance.instance_id}: Copy attempt {attempt + 1} failed: {e}')
            if attempt < max_retries:
                time.sleep(delay)
                continue
            else:
                logger.error(f'Instance {instance.instance_id}: Copy failed after {max_retries + 1} attempts: {e}')
                raise RuntimeError(f'Copy failed after {max_retries + 1} attempts: {e}')
    
    return False


def make_mkdir_safe(script_content: str) -> str:
    """
    将脚本中的 'mkdir ' 替换为 'mkdir -p '，但避免重复添加-p
    """
    # 只替换 "mkdir " 后面不是 "-p" 的情况
    # 使用负向前瞻确保mkdir后面不是-p开头
    result = re.sub(r'\bmkdir\s+(?!-p\b)', 'mkdir -p ', script_content)
    return result

def initialize_runtime(
    runtime: Runtime,
    instance: pd.Series,
):
    """Initialize the runtime for the agent.

    This function is called before the runtime is used to run the agent.
    """
    logger.info(f"{'-' * 50} BEGIN Runtime Initialization Fn {'-' * 50}")
    obs: CmdOutputObservation

    # Set instance id
    action = CmdRunAction(command='mkdir -p /workspace')
    logger.info(action, extra={'msg_type': 'ACTION'})
    run_action_with_retry(runtime, action, instance=instance)
    
    
    action = CmdRunAction(command='cd /workspace')
    logger.info(action, extra={'msg_type': 'ACTION'})
    run_action_with_retry(runtime, action, instance=instance)

    init_cmd = instance.init
    if init_cmd is not None:
        init_cmd = make_mkdir_safe(init_cmd)
        
        script_name = f'{instance.instance_id}_init.sh'

        with tempfile.TemporaryDirectory() as tmpdir:
            host_script_path = os.path.join(tmpdir, script_name)
            create_sh_file(host_script_path, init_cmd)
            # 使用带重试的copy方法
            copy_file_with_retry(instance, runtime, host_script_path, '/workspace')

        logger.info(f'Running init script: {script_name}')
        action = CmdRunAction(command=f'chmod +x ./{script_name} && ./{script_name}')
        logger.info(action, extra={'msg_type': 'ACTION'})
        run_action_with_retry(runtime, action, instance=instance)

    logger.info(f"{'-' * 50} END Runtime Initialization Fn {'-' * 50}")


def complete_runtime(
    runtime: Runtime,
    instance: pd.Series,
) -> dict[str, Any]:
    """Complete the runtime for the agent.

    This function is called before the runtime is used to run the agent.
    If you need to do something in the sandbox to get the correctness metric after
    the agent has run, modify this function.
    """
    logger.info(f"{'-' * 50} BEGIN Runtime Completion Fn {'-' * 50}")
    obs: CmdOutputObservation

    agent_answer = None
    get_agent_result_cmd = instance.get_agent_result
    if get_agent_result_cmd is not None:
        script_name = 'get_agent_result.sh'

        with tempfile.TemporaryDirectory() as tmpdir:
            host_script_path = os.path.join(tmpdir, script_name)
            create_sh_file(host_script_path, get_agent_result_cmd)
            copy_file_with_retry(instance, runtime, host_script_path, '/workspace')
            # runtime.copy_to(
            #     host_script_path,
            #     '/workspace',
            # )
            logger.info(f'Running get agent result cmd: {script_name}')

        action = CmdRunAction(
            command=f'chmod +x ./{script_name} && ./{script_name}',
        )
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = run_action_with_retry(runtime, action, instance=instance)
        agent_answer = obs.content
    # IF the agent answer is not found, retrieve it from the history
    # We wait until the controller finishes

    final_ans = None
    if instance.ground_truth is not None:
        final_ans = instance.ground_truth
    else:
        get_ground_truth_cmd = instance.get_ground_truth
        if get_ground_truth_cmd is not None:
            script_name = 'get_ground_truth.sh'
            with tempfile.TemporaryDirectory() as tmpdir:
                host_script_path = os.path.join(tmpdir, script_name)
                create_sh_file(host_script_path, get_ground_truth_cmd)
                copy_file_with_retry(instance, runtime, host_script_path, '/workspace')
                # runtime.copy_to(
                #     host_script_path,
                #     '/workspace',
                # )
            logger.info(f'Running get ground truth cmd: {script_name}')

            action = CmdRunAction(
                command=f'chmod +x ./{script_name} && ./{script_name}'
            )
            logger.info(action, extra={'msg_type': 'ACTION'})
            obs = run_action_with_retry(runtime, action, instance=instance)
            logger.info(obs, extra={'msg_type': 'OBSERVATION'})
            final_ans = obs.content

    logger.info(f"{'-' * 50} END Runtime Completion Fn {'-' * 50}")
    return {
        'final_ans': final_ans,
        'agent_answer': agent_answer,
    }

def action_transfer(solutions : list[dict]):
    action_list = []
    for solution in solutions:
        if 'observation' in solution:
            continue
        action = event_from_dict(solution)
        action_list.append(action)
    return action_list

def init_worker_process():
    """工作进程初始化函数"""
    # 每个进程创建一个事件循环
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # 注册进程退出时的清理函数
    import atexit
    
    def cleanup_on_exit():
        try:
            pending = asyncio.all_tasks(loop)
            if pending:
                for task in pending:
                    task.cancel()
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.close()
        except Exception as e:
            logger.error(f'Error during process cleanup: {e}')
    
    atexit.register(cleanup_on_exit)

# def solutions_generation(instance_datasets : list[pd.Series], n=N):
#     # 为多进程环境准备参数
#     tasks = []
#     for instance in instance_datasets:
#         for i in range(n):
#             tasks.append((instance, str(i)))
#     all_futures = []
#     future_to_task = {}
#     with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=init_worker_process) as executor:
#         for batch_start in range(0, len(tasks), BATCH_SIZE):
#             batch_end = min(batch_start + BATCH_SIZE, len(tasks))
#             batch_tasks = tasks[batch_start:batch_end]
#             for task in batch_tasks:
#                 future = executor.submit(solution_generation_parallel_wrapper, task)
#                 future_to_task[future] = task
#                 all_futures.append(future)
#             logger.info(f'Submitted {len(batch_tasks)} tasks for batch {batch_start//BATCH_SIZE+1}')
#             if batch_end < len(tasks):
#                 time.sleep(BATCH_DELAY)

#         for future in tqdm(concurrent.futures.as_completed(all_futures), desc='Solutions Generation', total=len(all_futures), dynamic_ncols=True):
#             task = future_to_task[future]
#             runtime_id = None
#             try:
#                 runtime_id = future.result()
#             except Exception as e:
#                 runtime_id = get_runtime_id(task[0].instance_id, task[1], None)
#             try:
#                 delete_container(runtime_id)
#             except Exception as e:
#                 logger.error(f'Runtime {runtime_id} Error deleting container: {e}')

def solutions_generation(instance_datasets: list[pd.Series], n=N):
    # 为多进程环境准备参数
    tasks = []
    for instance in instance_datasets:
        for i in range(n):
            tasks.append((instance, str(i)))
    
    total_tasks = len(tasks)
    pending_futures = {}
    
    # 创建两个进度条：一个显示提交进度，一个显示完成进度
    with tqdm(desc='Submitting Tasks', total=total_tasks, position=0, dynamic_ncols=True) as submit_pbar, \
         tqdm(desc='Completed Tasks', total=total_tasks, position=1, dynamic_ncols=True) as complete_pbar:
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=init_worker_process) as executor:
            submitted_count = 0
            
            # 分批提交任务
            for batch_start in range(0, len(tasks), BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, len(tasks))
                batch_tasks = tasks[batch_start:batch_end]
                
                # 提交当前批次的任务
                for task in batch_tasks:
                    future = executor.submit(solution_generation_parallel_wrapper, task)
                    pending_futures[future] = task
                    submitted_count += 1
                    submit_pbar.update(1)
                    submit_pbar.set_postfix({"Pending": len(pending_futures)})
                
                # 在提交过程中处理已完成的任务
                completed_futures = []
                for future in list(pending_futures.keys()):
                    if future.done():
                        completed_futures.append(future)
                
                # 处理完成的任务
                for future in completed_futures:
                    task = pending_futures.pop(future)
                    runtime_id = None
                    try:
                        runtime_id = future.result()
                    except Exception as e:
                        runtime_id = get_runtime_id(task[0].instance_id, task[1], None)
                        logger.error(f'Task failed: {e}')
                    
                    # 立即删除容器
                    try:
                        delete_container(runtime_id)
                    except Exception as e:
                        logger.error(f'Runtime {runtime_id} Error deleting container: {e}')
                    
                    complete_pbar.update(1)
                    complete_pbar.set_postfix({
                        "Pending": len(pending_futures),
                        "Runtime": runtime_id[-8:] if runtime_id else "Unknown"
                    })
                
                # 批次间延迟
                if batch_end < len(tasks):
                    time.sleep(BATCH_DELAY)
            
            # 提交完成，更新提交进度条状态
            submit_pbar.set_description("All Tasks Submitted")
            submit_pbar.close()
            
            while pending_futures:
                try:
                    # 使用 as_completed 等待任务完成，设置较短超时
                    for future in concurrent.futures.as_completed(pending_futures.keys(), timeout=1.0):
                        task = pending_futures.pop(future)
                        runtime_id = None
                        try:
                            runtime_id = future.result()
                        except Exception as e:
                            runtime_id = get_runtime_id(task[0].instance_id, task[1], None)
                            logger.error(f'Task failed: {e}')
                        
                        # 立即删除容器
                        try:
                            delete_container(runtime_id)
                        except Exception as e:
                            logger.error(f'Runtime {runtime_id} Error deleting container: {e}')
                        
                        complete_pbar.update(1)
                        complete_pbar.set_postfix({
                            "Pending": len(pending_futures),
                            "Runtime": runtime_id[-8:] if runtime_id else "Unknown"
                        })
                        
                        # 只处理一个完成的任务就跳出内循环，继续while循环
                        break
                        
                except concurrent.futures.TimeoutError:
                    # 超时是正常的，继续等待
                    continue
                except Exception as e:
                    logger.error(f'Unexpected error in task completion: {e}')
                    continue
                        


def solution_generation_parallel_wrapper(task_tuple):
    """包装函数，用于处理多进程参数传递"""
    instance, i = task_tuple
    runtime_id = solution_generation_parallel(instance, i)
    return runtime_id

def solution_generation_parallel(instance : pd.Series, i):
    # 由于多进程无法共享全局变量，需要在每个进程中重新初始化
    loop = asyncio.get_event_loop()
    try:
        runtime_id = loop.run_until_complete(run_single_solution_generation(instance, str(i)))
        return runtime_id
    except Exception as e:
        logger.error(f'Instance {instance.instance_id}/{i} Error in solution generation: {e}')
        logger.error(traceback.format_exc())
        raise e

async def run_single_solution_generation(instance : pd.Series, solution_id : str):
    # 在多进程中重新获取配置
    local_metadata = get_metadata(
        llm_config=get_llm_config(MODEL_NAME),
        dataset_name='AgentBench-OS',
        agent_class=OH_DEFAULT_AGENT,
        max_iterations=OH_MAX_ITERATIONS,
    )
    update_llm_config_for_completions_logging(local_metadata.llm_config, instance.file_store_path+f'/{instance.instance_id}', solution_id)
    config = get_config(local_metadata)
    config.file_store_path = instance.file_store_path+f'/{instance.instance_id}/solutions/{solution_id}'
    runtime_id = get_runtime_id(instance.instance_id, solution_id, None)
    runtime: Runtime = create_runtime(config, sid=runtime_id)
    await runtime.connect()

    try:
        initialize_runtime(runtime, instance=instance)
    except Exception as e:
        logger.error(f'Instance {instance.instance_id}/{solution_id} Error in initialize runtime: {e}')
    
    # Prepare instruction
    instruction = (
        f'Please fix the following issue.\n'
        'IMPORTANT: You should ONLY interact with the environment provided to you AND NEVER ASK FOR HUMAN HELP.\n'
        'Please encapsulate your final answer (answer ONLY) within <solution> and </solution>.\n'
        'For example: The answer to the question is <solution> 42 </solution>.\n'
        '# Problem \n'
        f'{instance.description}\n\n'
    )
    instruction += (
        'IMPORTANT: You should ONLY interact with the environment provided '
        'to you AND NEVER ASK FOR HUMAN HELP.\n'
    )
    # NOTE: You can actually set slightly different instruction for different agents
    instruction += INST_SUFFIXES[local_metadata.agent_class]
    # Run the agent
    state: State | None = await run_controller(
        sid=solution_id,
        config=config,
        initial_user_action=MessageAction(content=instruction),
        runtime=runtime,
        fake_user_response_fn=FAKE_RESPONSES[local_metadata.agent_class],
    )
    if state is None:
        raise ValueError('State should not be None.')
    return runtime_id

def make_prompt(task : str, **kwargs):
    if task == 'testing':
        if kwargs.get('type', None) == 'QA':
            from prompts.testing import system_prompt, qa_instruction
            return [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': qa_instruction.format(description=kwargs['description'], init=kwargs['init'])}]
        elif kwargs.get('type', None) == 'EXEC':
            from prompts.testing import system_prompt, execution_instruction
            return [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': execution_instruction.format(description=kwargs['description'], init=kwargs['init'])}]
        else:
            raise ValueError(f'Invalid task type: {task}')
    elif task == 'type':
        from prompts.type import system_prompt, type_instruction
        return [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': type_instruction.format(query=kwargs['query'])}]
    else:
        raise ValueError(f'Invalid task: {task}')

def llm_call(prompt: list[dict[str, str]]):
    llm_config = metadata.llm_config

    kwargs: dict[str, Any] = {
            'temperature': llm_config.temperature,
            'max_completion_tokens': llm_config.max_output_tokens,
        }
    response = completion(
            model=llm_config.model,
            messages=prompt,
            api_key=llm_config.api_key.get_secret_value() if llm_config.api_key else None,
            base_url=llm_config.base_url,
            api_version=llm_config.api_version,
            custom_llm_provider=llm_config.custom_llm_provider,
            timeout=llm_config.timeout,
            top_p=llm_config.top_p,
            drop_params=llm_config.drop_params,
            client=client,
            **kwargs,
        )
    return response

def testing_generation_improved(instance_datasets: list[pd.Series], n=N):
    total_tasks = len(instance_datasets) * n
    pending_futures = {}
    
    with tqdm(desc='Submitting Tasks', total=total_tasks, position=0, dynamic_ncols=True) as submit_pbar, \
         tqdm(desc='Completed Tasks', total=total_tasks, position=1, dynamic_ncols=True) as complete_pbar:
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            submitted_count = 0
            
            # 分批提交任务，避免一次性创建太多futures
            for batch_start in range(0, len(instance_datasets), BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, len(instance_datasets))
                batch_instances = instance_datasets[batch_start:batch_end]
                
                # 提交当前批次的任务
                for instance in batch_instances:
                    for i in range(n):
                        future = executor.submit(testing_generation_parallel, instance, str(i))
                        pending_futures[future] = (instance, i)
                        submitted_count += 1
                        submit_pbar.update(1)
                        submit_pbar.set_postfix({"Pending": len(pending_futures)})
                
                # 在提交过程中处理已完成的任务
                completed_futures = []
                for future in list(pending_futures.keys()):
                    if future.done():
                        completed_futures.append(future)
                
                # 处理完成的任务并立即从pending_futures中移除
                for future in completed_futures:
                    try:
                        result = future.result()
                        del pending_futures[future]  # 立即删除，释放内存
                        complete_pbar.update(1)
                    except Exception as e:
                        logger.error(f'Task failed: {e}')
                        del pending_futures[future]
                        complete_pbar.update(1)
                
                # 批次间延迟
                if batch_end < len(instance_datasets):
                    time.sleep(BATCH_DELAY)
            
            # 处理剩余的pending futures
            while pending_futures:
                try:
                    # 使用短超时的as_completed，避免长时间阻塞
                    for future in concurrent.futures.as_completed(pending_futures.keys(), timeout=1.0):
                        try:
                            result = future.result()
                            del pending_futures[future]
                            complete_pbar.update(1)
                        except Exception as e:
                            logger.error(f'Task failed: {e}')
                            del pending_futures[future]
                            complete_pbar.update(1)
                        break  # 只处理一个就跳出，继续while循环
                        
                except concurrent.futures.TimeoutError:
                    continue  # 超时是正常的，继续等待
                
def testing_generation(instance_datasets : list[pd.Series], n=N):
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = []
        for id, instance in tqdm(enumerate(instance_datasets), desc='Testing Generation submitting', total=len(instance_datasets), dynamic_ncols=True):
            for i in range(n):
                futures.append(executor.submit(testing_generation_parallel, instance, str(i)))

        for future in tqdm(concurrent.futures.as_completed(futures), desc='Testing Generation', total=len(futures), dynamic_ncols=True):
            future.result()

def testing_generation_parallel(instance : pd.Series, i):
    dir_path = instance.file_store_path+f'/{instance.instance_id}/testings/'
    os.makedirs(dir_path, exist_ok=True)
    if os.path.exists(dir_path+f'{i}.sh'):
        return None
    prompt = make_prompt('testing', type=instance.type, description=instance.description, init=instance.init)
    # 增加llm_call的重试机制，最多重试3次，每次失败后等待2秒
    max_retry = 3
    for attempt in range(max_retry):
        try:
            res = llm_call(prompt)
            break
        except Exception as e:
            logger.warning(f"llm_call 第{attempt+1}次调用失败: {e}")
            if attempt < max_retry - 1:
                time.sleep(2)
            else:
                logger.error(f'Testing Generation Error: {e}')
                raise

    if res['choices'][0]['message']['role'] == 'assistant':
        content = res['choices'][0]['message']['content']
        testing = re.findall(r'<testing>(.*?)</testing>', content, re.DOTALL)
        if len(testing) == 0:
            logger.warning(f'Failed to parse model answer: {content}')
            testing = content
        else:
            testing = testing[0]    

    with open(dir_path+f'{i}.sh', 'w', encoding='utf-8') as f:
        f.write(testing.replace('\r\n', '\n'))
    os.chmod(dir_path+f'{i}.sh', 0o755)
    return testing

def type_check(instance : pd.Series):
    prompt = make_prompt('type', query=instance.description)
    response = llm_call(prompt)
    if response['choices'][0]['message']['role'] == 'assistant':
        content = response['choices'][0]['message']['content']
        type = re.findall(r'<type>(.*?)</type>', content, re.DOTALL)
        while len(type) == 0:
            response = llm_call(prompt)
            content = response['choices'][0]['message']['content']
            type = re.findall(r'<type>(.*?)</type>', content, re.DOTALL)
        instance.type = type[0].strip()
    else:
        logger.warning(f'Failed to parse model answer: {response}')
        raise ValueError(f'Failed to parse model answer: {response}')

def verify(instance : pd.Series, solution_id : str, testing_id : str):
    try:
        return _verify(instance, solution_id, testing_id)
    except Exception as e:
        logger.error(f'Error in verify: {e}')
        logger.error(traceback.format_exc())
        return {
            'instance_id': instance.instance_id,
            'instance': instance.to_dict(),
            'instruction': instance.description,
            'solution_id': solution_id,
            'testing_id': testing_id,
            'metadata': metadata.model_dump_json(),
            'history': [],
            'test_result': {
                'agent_answer': 'UNKNOWN_ERROR',
                'final_answer': 'UNKNOWN_ERROR',
                'check_method': instance.comparison_method,
                'result': False,
            },
        }

def _verify(instance : pd.Series, solution_id : str, testing_id : str):
    """
    verify the solution if it can pass the testing.
    """
    local_metadata = get_metadata(
        llm_config=get_llm_config(MODEL_NAME),
        dataset_name='AgentBench-OS',
        agent_class=OH_DEFAULT_AGENT,
        max_iterations=OH_MAX_ITERATIONS,
    )
    config = get_config(local_metadata)
    solution_path = instance.file_store_path+f'/{instance.instance_id}/solutions/{solution_id}/sessions/{get_runtime_id(instance.instance_id, solution_id, None)}/events/'

    def get_solution():
        solution = []
        if not os.path.exists(solution_path):
            return solution
        solution_files = [f for f in os.listdir(solution_path) if os.path.isfile(os.path.join(solution_path, f))]
        for i in range(len(solution_files)):
            with open(solution_path+f'/{i}.json', 'r') as f:
                solution.append(json.load(f))
        solution = action_transfer(solution)
        return solution
    solution = get_solution()

    if len(solution) == 0:
        return {
            'instance_id': instance.instance_id,
            'instance': instance.to_dict(),
            'instruction': instance.description,
            'solution_id': solution_id,
            'testing_id': testing_id,
            'metadata': metadata.model_dump_json(),
            'history': [],
            'test_result': {
                'agent_answer': 'NO_SOLUTION_FOUND',
                'final_answer': 'NO_SOLUTION_FOUND',
                'check_method': instance.comparison_method,
                'result': False,
            },
        }
    if testing_id != 'ground_truth' and not os.path.exists(instance.file_store_path+f'/{instance.instance_id}/testings/{testing_id}.sh'):
        return {
            'instance_id': instance.instance_id,
            'instance': instance.to_dict(),
            'instruction': instance.description,
            'solution_id': solution_id,
            'testing_id': testing_id,
            'metadata': metadata.model_dump_json(),
            'history': [],
            'test_result': {
                'agent_answer': 'NO_TESTING_FOUND',
                'final_answer': 'NO_TESTING_FOUND',
                'check_method': instance.comparison_method,
                'result': False,    
            },
        }
    
    # =============================================
    # create sandbox and run the agent
    # =============================================
    runtime: Runtime = create_runtime(config, sid=get_runtime_id(instance.instance_id, solution_id, testing_id))
    call_async_from_sync(runtime.connect)
    try:
        initialize_runtime(runtime, instance=instance)
    except Exception as e:
        logger.error(f'Instance {instance.instance_id}/{solution_id} Error in initialize runtime: {e}')
        raise e

    # =============================================
    # result evaluation
    # =============================================
    agent_answer = None
    final_ans = None
    if testing_id is not None and testing_id != 'ground_truth':
        testing_path = instance.file_store_path+f'/{instance.instance_id}/testings/{testing_id}.sh'
        script_name = f'{testing_id}.sh'
        if instance.type == 'QA':
            for event in reversed(solution):
                if event.source == 'agent':
                    if isinstance(event, AgentFinishAction):
                        raw_ans = event.thought
                        break
                    elif isinstance(event, MessageAction):
                        raw_ans = event.content
                        break
                    elif isinstance(event, CmdRunAction):
                        raw_ans = event.thought
                        break
            agent_answer = re.findall(r'<solution>(.*?)</solution>', raw_ans, re.DOTALL)
            if len(agent_answer) == 0:
                logger.warning(f'Failed to parse model answer: {raw_ans}')
                agent_answer = raw_ans
            else:
                agent_answer = agent_answer[0]
            runtime.copy_to(
                    testing_path,
                    '/workspace',
                )
            logger.info(f'Running testing cmd: {testing_path}')
            action = CmdRunAction(
                command=f'chmod +x ./{script_name} && ./{script_name}',
            )
            logger.info(action, extra={'msg_type': 'ACTION'})
            obs = runtime.run_action(action)
            logger.info(obs, extra={'msg_type': 'OBSERVATION'})
            # assert obs.exit_code == 0
            final_ans = obs.content
            logger.info(f'Final message: {agent_answer} | Ground truth: {final_ans} | Comparison method: {instance.comparison_method}')
            test_result = compare_results(instance.comparison_method, agent_answer, final_ans)
        elif instance.type == 'EXEC':
            for action in solution:
                if action.source == 'agent':
                    logger.info(action, extra={'msg_type': 'ACTION'})
                    obs = runtime.run_action(action)
                    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
            runtime.copy_to(
                    testing_path,
                    '/workspace',
                )
            logger.info(f'Running testing cmd: {testing_path}')
            action = CmdRunAction(
                command=f'chmod +x ./{script_name} && ./{script_name}',
            )
            logger.info(action, extra={'msg_type': 'ACTION'})
            obs = runtime.run_action(action)
            logger.info(obs, extra={'msg_type': 'OBSERVATION'})
            # assert obs.exit_code == 0
            final_ans = obs.content
            logger.info(f'Final message: {agent_answer} | Ground truth: {final_ans} | Comparison method: {instance.comparison_method}')
            test_result = (final_ans == 'true')
    else:
        for action in solution:
            if action.source == 'agent':
                logger.info(action, extra={'msg_type': 'ACTION'})
                obs = runtime.run_action(action)
                logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        try:
            return_val = complete_runtime(runtime, instance)
        except Exception as e:
            logger.error(f'Instance {instance.instance_id}/{solution_id} Error in complete runtime: {e}')
            return_val = {
                'agent_answer': 'UNKNOWN_ERROR',
                'final_ans': 'UNKNOWN_ERROR',
            }
        agent_answer = return_val['agent_answer']
        final_ans = return_val['final_ans']
        if agent_answer is None:
            agent_answer = ''
            logger.info('Retrieving agent answer from history.')
            raw_ans = ''

            # retrieve the last agent message or thought
            for event in reversed(solution):
                if event.source == 'agent':
                    if isinstance(event, AgentFinishAction):
                        raw_ans = event.thought
                        break
                    elif isinstance(event, MessageAction):
                        raw_ans = event.content
                        break
                    elif isinstance(event, CmdRunAction):
                        raw_ans = event.thought
                        break

            # parse the answer for a solution tag
            agent_answer = re.findall(r'<solution>(.*?)</solution>', raw_ans, re.DOTALL)
            if len(agent_answer) == 0:
                logger.warning(f'Failed to parse model answer: {raw_ans}')
                agent_answer = raw_ans
            else:
                agent_answer = agent_answer[0]

        comparison_method = instance.comparison_method
        logger.info(
            f'Final message: {agent_answer} | Ground truth: {final_ans} | Comparison method: {comparison_method}'
        )
        test_result = compare_results(comparison_method, agent_answer, final_ans)


    # history is now available as a stream of events, rather than list of pairs of (Action, Observation)
    # for compatibility with the existing output format, we can remake the pairs here
    # remove when it becomes unnecessary
    histories = compatibility_for_eval_history_pairs(solution)

    # Save the output
    output = {
        'instance_id': instance.instance_id,
        'instance': instance.to_dict(),
        'instruction': instance.description,
        'solution_id': solution_id,
        'testing_id': testing_id,
        'metadata': metadata.model_dump_json(),
        'history': histories,
        'test_result': {
            'agent_answer': agent_answer,
            'final_answer': final_ans,
            'check_method': instance.comparison_method,
            'result': test_result,
        },
    }
    return output



def final_check(instance : pd.Series, sort_solution_id_list : list[str], sort_testing_id_list : list[str]):
    save_path = instance.file_store_path+f'/{instance.instance_id}/final_results.json'
    correct_solution_id = sort_solution_id_list[0]
    correct_testing_id = sort_testing_id_list[0]
    wrong_solution_id = sort_solution_id_list[-1]
    positive_output = None
    negative_output = None
    ground_truth_positive_output = None
    ground_truth_negative_output = None
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        ground_truth_positive_future = executor.submit(verify, instance, correct_solution_id, 'ground_truth')
        ground_truth_negative_future = executor.submit(verify, instance, wrong_solution_id, 'ground_truth')
        positive_future = executor.submit(verify, instance, correct_solution_id, correct_testing_id)
        negative_future = executor.submit(verify, instance, wrong_solution_id, correct_testing_id)
        positive_output = positive_future.result()
        negative_output = negative_future.result()
        ground_truth_positive_output = ground_truth_positive_future.result()
        ground_truth_negative_output = ground_truth_negative_future.result()
    positive_result = positive_output['test_result']['result']
    negative_result = negative_output['test_result']['result']
    ground_truth_positive_result = ground_truth_positive_output['test_result']['result']
    ground_truth_negative_result = ground_truth_negative_output['test_result']['result']
    
    if 'UNKNOWN_ERROR' in [ground_truth_positive_output['test_result']['agent_answer'], ground_truth_negative_output['test_result']['agent_answer'], positive_output['test_result']['agent_answer'], negative_output['test_result']['agent_answer']]:
        output = {
            'instance_id': instance.instance_id,
            'instance': instance.to_dict(),
            'instruction': instance.description,
            'correct_solution_id': correct_solution_id,
            'correct_testing_id': correct_testing_id,
            'wrong_solution_id': wrong_solution_id,
            'ground_truth_positive_result': ground_truth_positive_result,
            'ground_truth_negative_result': ground_truth_negative_result,
            'positive_result': positive_result,
            'negative_result': negative_result,
            'result': 'UNKNOWN_ERROR',
        }
    else:

        output = {
            'instance_id': instance.instance_id,
            'instance': instance.to_dict(),
            'instruction': instance.description,
            'correct_solution_id': correct_solution_id,
            'correct_testing_id': correct_testing_id,
            'wrong_solution_id': wrong_solution_id,
            'ground_truth_positive_result': ground_truth_positive_result,
            'ground_truth_negative_result': ground_truth_negative_result,
            'positive_result': positive_result,
            'negative_result': negative_result,
            'result': ground_truth_positive_future and positive_future and (ground_truth_negative_result == negative_result),
        }
    print(output)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4)
    return output

def generation_check(instance : pd.Series, sort_solution_id_list : list[str], sort_testing_id_list : list[str]):
    save_path = instance.file_store_path+f'/{instance.instance_id}/gen_final_results.json'
    correct_solution_id = sort_solution_id_list[0]
    correct_testing_id = sort_testing_id_list[0]
    positive_output = verify(instance, correct_solution_id, correct_testing_id)
    positive_result = positive_output['test_result']['result']
    
    if 'UNKNOWN_ERROR' in [positive_output['test_result']['agent_answer']]:
        output = {
            'instance_id': instance.instance_id,
            'instance': instance.to_dict(),
            'instruction': instance.description,
            'correct_solution_id': correct_solution_id,
            'correct_testing_id': correct_testing_id,
            'positive_result': positive_result,
            'result': 'UNKNOWN_ERROR',
        }
    else:

        output = {
            'instance_id': instance.instance_id,
            'instance': instance.to_dict(),
            'instruction': instance.description,
            'correct_solution_id': correct_solution_id,
            'correct_testing_id': correct_testing_id,
            'positive_result': positive_result,
            'result': positive_result,
        }

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4)
    return output


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

def get_our_container_count(prefix="openhands-runtime-"):
    """获取我们创建的容器数量"""
    try:
        client = docker.from_env(timeout=5)
        containers = client.containers.list(all=True)
        our_containers = [c for c in containers if c.name.startswith(prefix)]
        count = len(our_containers)
        client.close()
        return count
    except Exception as e:
        logger.error(f"Failed to get container count: {e}")
        return 512  # 返回一个大数，暂停任务提交

def controlled_strategy_computation(check_datasets, program_path, max_containers=128, max_concurrent=2):
    """控制并发和容器数量的策略计算"""
    
    final_check_datasets = []
    pending_tasks = list(check_datasets)
    
    logger.info(f"Starting controlled strategy computation: "
               f"max_containers={max_containers}, max_concurrent={max_concurrent}")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_concurrent) as executor:
        running_futures = {}
        
        with tqdm(total=len(check_datasets), desc='Strategy computation (controlled)', dynamic_ncols=True) as pbar:
            
            while pending_tasks or running_futures:
                
                # 启动新任务（如果容器数量允许且有空闲槽位）
                while (len(running_futures) < max_concurrent and 
                       pending_tasks):
                    
                    # 检查容器数量
                    current_containers = get_our_container_count()
                    if current_containers >= max_containers:
                        logger.info(f"Container limit reached ({current_containers}/{max_containers}), "
                                   f"waiting before starting new tasks...")
                        break
                    
                    # 启动新任务
                    instance = pending_tasks.pop(0)
                    solution_id_list = testing_id_list = [str(i) for i in range(N)]
                    future = executor.submit(strategy_wrapper, program_path, instance, solution_id_list, testing_id_list)
                    running_futures[future] = instance
                    
                    logger.info(f"Started strategy computation for {instance.instance_id} "
                               f"(containers: {current_containers}, concurrent: {len(running_futures)})")
                
                # 检查完成的任务
                if running_futures:
                    completed_futures = []
                    for future in running_futures:
                        if future.done():
                            completed_futures.append(future)
                    
                    # 处理完成的任务
                    for future in completed_futures:
                        instance = running_futures.pop(future)
                        try:
                            result = future.result()
                            sort_solution_id_list, sort_testing_id_list = result
                            with open(f'{instance.file_store_path}/{instance.instance_id}/sort_list.json', 'w', encoding='utf-8') as f:
                                json.dump({'solution_id_list': sort_solution_id_list, 'testing_id_list': sort_testing_id_list}, f, indent=4)
                            final_check_datasets.append((instance, sort_solution_id_list, sort_testing_id_list))
                            
                            pbar.update(1)
                            pbar.set_postfix({
                                'Instance': instance.instance_id[:10],
                                'Containers': get_our_container_count(),
                                'Running': len(running_futures),
                                'Pending': len(pending_tasks)
                            })
                            
                        except Exception as e:
                            logger.error(f"Strategy computation failed for {instance.instance_id}: {e}")
                            pbar.update(1)
                
                # 如果没有完成的任务，稍等一下
                if running_futures and not completed_futures:
                    time.sleep(40)
    
    logger.info(f"Controlled strategy computation completed. Processed {len(final_check_datasets)} instances.")
    return final_check_datasets

def evaluate(program_path : str):
    """
    Evaluate the program by running it once and checking the sum of radii

    Args:
        program_path: Path to the program file

    Returns:
        Dictionary of metrics
    """ 
    current_time = datetime.datetime.now()
    FILE_STORE_PATH = 'verifier_file_store' + f'/{current_time.strftime("%Y%m%d%H%M%S")}'
    dataset = load_dataset('iFurySt/AgentBench')
    agent_bench_tests = dataset['osbench'].to_pandas()
    new_datasets = []
    for _, instance in agent_bench_tests.iterrows():
        instance = instance.copy()
        instance['type'] = None
        instance['file_store_path'] = FILE_STORE_PATH
        instance['idx'] = len(new_datasets)
        new_datasets.append(instance)
        # break
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
    check_datasets = valid_instances
    solutions_generation(check_datasets)
    check_datasets = [instance for instance in check_datasets if instance.type != 'WRONG']
    testing_generation(check_datasets)
    
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    
    final_check_datasets = []

    for instance in tqdm(check_datasets, desc='verifying', total=len(check_datasets), dynamic_ncols=True):
        solution_id_list = testing_id_list = [str(i) for i in range(N)]
        sort_solution_id_list, sort_testing_id_list = strategy_wrapper(program_path, instance, solution_id_list, testing_id_list)
        final_check_datasets.append((instance, sort_solution_id_list, sort_testing_id_list))

            
    final_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS//2) as executor:
        futures = [executor.submit(final_check, instance, solution_id_list, testing_id_list) for instance, solution_id_list, testing_id_list in final_check_datasets]
        for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), desc='final checking', total=len(futures), dynamic_ncols=True):
            output = future.result()
            try:
                delete_container(get_runtime_id(output['instance_id'], output['correct_solution_id'], 'ground_truth'))
                delete_container(get_runtime_id(output['instance_id'], output['wrong_solution_id'], output['correct_testing_id']))
                delete_container(get_runtime_id(output['instance_id'], output['correct_solution_id'], output['correct_testing_id']))
                delete_container(get_runtime_id(output['instance_id'], output['wrong_solution_id'], 'ground_truth'))
            except Exception as e:
                logger.error(f'Failed to delete container: {e}')
            result = output['result']
            if result == 'UNKNOWN_ERROR':
                continue
            final_results.append(result)
            
    final_score = 0
    for result in final_results:
        if result:
            final_score += 1
    print(f'Final score: {final_score}/{len(final_results)}')
    return {"score": final_score/len(final_results)}


def generation(dataset_path : str, strategy_path : str):
    current_time = datetime.datetime.now()
    FILE_STORE_PATH = 'verifier_data/file_store' + '/20250813133246'
    
    new_datasets = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            instance = json.loads(line)
            instance['file_store_path'] = FILE_STORE_PATH
            new_datasets.append(pd.Series(instance))
            
    # tmp_new_datasets = new_datasets[1840:]
    # solutions_generation(tmp_new_datasets)
    check_datasets = new_datasets[:952] + new_datasets[1000:1250]
    # testing_generation(check_datasets)
    # testing_generation_improved(check_datasets)
    
    final_check_datasets = []
        
    for instance in tqdm(check_datasets, desc='verifying', total=len(check_datasets), dynamic_ncols=True):
        if os.path.exists(f'{instance.file_store_path}/{instance.instance_id}/sort_list.json'):
            data = json.load(open(f'{instance.file_store_path}/{instance.instance_id}/sort_list.json', 'r', encoding='utf-8'))
            sort_solution_id_list = data['solution_id_list']
            sort_testing_id_list = data['testing_id_list']
        else:
            solution_id_list = testing_id_list = [str(i) for i in range(N)]
            sort_solution_id_list, sort_testing_id_list = strategy_wrapper(strategy_path, instance, solution_id_list, testing_id_list)
            with open(f'{instance.file_store_path}/{instance.instance_id}/sort_list.json', 'w', encoding='utf-8') as f:
                json.dump({'solution_id_list': sort_solution_id_list, 'testing_id_list': sort_testing_id_list}, f, indent=4)
        final_check_datasets.append((instance, sort_solution_id_list, sort_testing_id_list))
        
    with concurrent.futures.ProcessPoolExecutor(max_workers=256) as executor:
        futures = [executor.submit(generation_check, instance, sort_solution_id_list, sort_testing_id_list) for instance, sort_solution_id_list, sort_testing_id_list in final_check_datasets]
        
        # 统计变量
        completed_count = 0
        success_count = 0
        error_count = 0
        
        # 创建进度条
        pbar = tqdm(total=len(futures), desc='Generation Checking', dynamic_ncols=True)
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result['result'] != 'UNKNOWN_ERROR':
                    success_count += 1
                else:
                    error_count += 1
                pbar.set_postfix({
                    '成功': success_count,
                    '错误': error_count,
                    '成功率': f'{success_count/(completed_count+1)*100:.1f}%'
                })
                try:
                    delete_container(get_runtime_id(result['instance_id'], result['correct_solution_id'], result['correct_testing_id']))
                except Exception as e:
                    logger.error(f'Failed to delete container: {e}')
            except Exception as e:
                error_count += 1
                pbar.set_postfix({
                    '成功': success_count,
                    '错误': error_count,
                    '成功率': f'{success_count/(completed_count+1)*100:.1f}%' if completed_count > 0 else '0.0%'
                })
                # 可选：记录错误日志
                print(f"\n警告: 任务执行失败 - {str(e)}")
            finally:
                completed_count += 1
                pbar.update(1)
        
        pbar.close()
        
        # 输出最终统计信息
        print(f"\n生成检查完成:")
        print(f"  总任务数: {len(futures)}")
        print(f"  成功: {success_count}")
        print(f"  失败: {error_count}")
        print(f"  成功率: {success_count/len(futures)*100:.1f}%")

def run_infer():
    current_time = datetime.datetime.now()
    FILE_STORE_PATH = 'verifier_file_store' + f'/{current_time.strftime("%Y%m%d%H%M%S")}'
    dataset = load_dataset('iFurySt/AgentBench')
    agent_bench_tests = dataset['osbench'].to_pandas()
    new_datasets = []
    for _, instance in agent_bench_tests.iterrows():
        instance = instance.copy()
        instance['type'] = None
        instance['file_store_path'] = FILE_STORE_PATH
        instance['idx'] = len(new_datasets)
        new_datasets.append(instance)
        # break
    check_datasets = new_datasets
    
    solutions_generation(check_datasets)
    
    results = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(verify, instance, "0", 'ground_truth') for instance in check_datasets]
        for future in tqdm(concurrent.futures.as_completed(futures), desc='verifying', total=len(futures), dynamic_ncols=True):
            result = future.result()
            if result["test_result"]["agent_answer"] == "UNKNOWN_ERROR":
                continue
            results.append(result)

            try:
                delete_container(get_runtime_id(result['instance_id'], result['solution_id'], 'ground_truth'))
            except Exception as e:
                logger.error(f'Failed to delete container: {e}')

    score = []
    for result in results:
        if result["test_result"]["result"] is True:
            score.append(1)
        else:
            score.append(0)
    print(f'Final score: {sum(score)}/{len(score)}')
    
    with open(os.path.join('verifier', f'{MODEL_NAME}_results.jsonl'), 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

if __name__ == '__main__':
    evaluate('verifier/openevolve_output/checkpoints/checkpoint_7/best_program.py')