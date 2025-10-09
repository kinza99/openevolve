import json
import logging
import multiprocessing as mp
import os
import pathlib
import signal
import subprocess
import time
import traceback
from contextlib import contextmanager
from inspect import signature
from typing import Any, Awaitable, Callable, TextIO

import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from openhands.controller.state.state import State
from openhands.core.config import LLMConfig, SandboxConfig
from openhands.core.config.agent_config import AgentConfig
from openhands.core.config.condenser_config import (
    CondenserConfig,
    NoOpCondenserConfig,
)
from openhands.core.exceptions import (
    AgentRuntimeBuildError,
    AgentRuntimeDisconnectedError,
    AgentRuntimeError,
    AgentRuntimeNotFoundError,
    AgentRuntimeNotReadyError,
    AgentRuntimeTimeoutError,
    AgentRuntimeUnavailableError,
)
from openhands.core.logger import get_console_handler
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import Action
from openhands.events.action.message import MessageAction
from openhands.events.event import Event
from openhands.events.serialization.event import event_to_dict
from openhands.events.utils import get_pairs_from_events
from openhands.memory.condenser import get_condensation_metadata
import docker

CONTAINER_NAME_PREFIX = 'openhands-runtime-'

def get_runtime_id(instance_id : str, solution_id : str, testing_id : str | None):
    if testing_id is None:
        return f'openhands-{instance_id.replace(".","_")}-{solution_id}'
    else:
        if testing_id == 'ground_truth':
            return f'openhands-{instance_id.replace(".","_")}-{solution_id}-ground_truth'
        else:
            return f'openhands-{instance_id.replace(".","_")}-{solution_id}-{testing_id}'

import concurrent.futures
import threading
from typing import Optional

# 创建一个全局的线程池用于异步删除容器
_container_deletion_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
_executor_lock = threading.Lock()

def get_container_deletion_executor():
    """获取用于异步删除容器的线程池"""
    global _container_deletion_executor
    if _container_deletion_executor is None:
        with _executor_lock:
            if _container_deletion_executor is None:
                _container_deletion_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=256,  # 限制并发删除的数量
                    thread_name_prefix="container-deletion"
                )
    return _container_deletion_executor

def delete_container_async(
    name: str,
    force: bool = True,
    dry_run: bool = False
) -> concurrent.futures.Future:
    """
    异步删除 Docker 容器
    
    参数:
        name: 容器名称
        force: 是否强制删除 (即使容器正在运行)
        dry_run: 如果为 True，只打印要删除的容器而不实际删除
    
    返回:
        Future 对象，可以用来检查删除状态或获取结果
    """
    executor = get_container_deletion_executor()
    return executor.submit(delete_container, name, force, dry_run)

def delete_container_fire_and_forget(
    name: str,
    force: bool = True,
    dry_run: bool = False
):
    """
    异步删除容器，不等待结果（fire-and-forget 模式）
    
    参数:
        name: 容器名称
        force: 是否强制删除
        dry_run: 是否为试运行模式
    """
    def deletion_callback(future):
        """删除完成后的回调函数"""
        try:
            future.result()  # 这会抛出任何在删除过程中发生的异常
            logger.info(f'Container {name} deleted successfully in background')
        except Exception as e:
            logger.error(f'Failed to delete container {name} in background: {e}')
    
    future = delete_container_async(name, force, dry_run)
    future.add_done_callback(deletion_callback)
    return future

def delete_all_stopped_containers(
    force: bool = True,
    dry_run: bool = False,
    prefix_filter: str = CONTAINER_NAME_PREFIX
):
    """
    删除所有停止状态的 Docker 容器

    参数:
        force: 是否强制删除
        dry_run: 如果为 True，只打印要删除的容器而不实际删除
        prefix_filter: 可选的容器名称前缀过滤器

    返回:
        已删除容器的数量
    """
    kwargs = {
        'timeout': 60,
    }
    client = docker.from_env(**kwargs)
    
    try:
        # 获取所有容器
        containers = client.containers.list(all=True)

        # 筛选停止状态的容器
        stopped_containers = [
            c for c in containers 
            if c.status in ['exited', 'stopped']
        ]
        
        # 如果有前缀过滤器，进一步筛选
        if prefix_filter:
            stopped_containers = [
                c for c in stopped_containers 
                if c.name.startswith(prefix_filter)
            ]

        logger.info(f"找到 {len(stopped_containers)} 个停止状态的容器")
        
        deleted_count = 0
        for container in stopped_containers:
            container_info = {
                "id": container.id,
                "name": container.name,
                "status": container.status
            }

            if dry_run:
                logger.info(f"将删除容器: {container_info}")
                continue

            try:
                logger.info(f"正在删除容器: {container_info}")
                container.remove(force=force)
                logger.info(f"已成功删除容器: {container.name}")
                deleted_count += 1
            except Exception as e:
                logger.error(f"删除容器 {container.name} 失败: {str(e)}")

        return deleted_count
        
    except Exception as e:
        logger.error(f"获取容器列表失败: {str(e)}")
        return 0
    finally:
        client.close()

def delete_container(
    name: str,
    force: bool = True,
    dry_run: bool = False
):
    """
    删除名称以特定字符串结尾的 Docker 容器

    参数:
        suffix: 容器名称的结尾字符串
        force: 是否强制删除 (即使容器正在运行)
        dry_run: 如果为 True，只打印要删除的容器而不实际删除

    返回:
        已删除容器的 ID 列表
    """
    kwargs = {
        'timeout': 600,
    }
    client = docker.from_env(**kwargs)
    containers = client.containers.list(all=True)
    if name == 'all':
        filtered_containers = [
            c for c in containers if c.name.startswith(CONTAINER_NAME_PREFIX)
        ]
    else:
        # 筛选符合条件的容器
        filtered_containers = [
            c for c in containers if c.name == (CONTAINER_NAME_PREFIX + name)
        ]
    try:
        for container in filtered_containers:
            container_info = {
                "id": container.id,
                "name": container.name,
                "status": container.status
            }

            if dry_run:
                logger.info(f"将删除容器: {container_info}")
                continue

            try:
                logger.info(f"正在删除容器: {container_info}")
                container.remove(force=force)
                logger.info(f"已成功删除容器: {container.name}")
            except Exception as e:
                logger.error(f"删除容器 {container.name} 失败: {str(e)}")
    finally:
        client.close()

class EvalMetadata(BaseModel):
    agent_class: str
    llm_config: LLMConfig
    agent_config: AgentConfig | None = None
    max_iterations: int
    eval_output_dir: str | None = None
    eval_workspace_path: str | None = None
    start_time: str | None = None
    git_commit: str | None = None
    dataset: str | None = None
    data_split: str | None = None
    details: dict[str, Any] | None = None
    condenser_config: CondenserConfig | None = None


class EvalOutput(BaseModel):
    # NOTE: User-specified
    instance_id: str
    # output of the evaluation
    # store anything that is needed for the score calculation
    test_result: dict[str, Any]

    instruction: str | None = None

    # Interaction info
    metadata: EvalMetadata | None = None
    # list[tuple[dict[str, Any], dict[str, Any]]] - for compatibility with the old format
    history: (
        list[dict[str, Any]] | list[tuple[dict[str, Any], dict[str, Any]]] | None
    ) = None
    metrics: dict[str, Any] | None = None
    error: str | None = None

    # Optionally save the input test instance
    instance: dict[str, Any] | None = None


class EvalException(Exception):
    pass


class EvalTimeoutException(Exception):
    pass


@contextmanager
def timeout(seconds: int):
    def timeout_handler(signum, frame):
        raise EvalTimeoutException(f'Function timed out after {seconds} seconds')

    # Set up the signal handler
    original_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Restore the original handler and disable the alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)


def codeact_user_response(
    state: State,
    encapsulate_solution: bool = False,
    try_parse: Callable[[Action], str] | None = None,
) -> str:
    encaps_str = (
        (
            'Please encapsulate your final answer (answer ONLY) within <solution> and </solution>.\n'
            'For example: The answer to the question is <solution> 42 </solution>.\n'
        )
        if encapsulate_solution
        else ''
    )
    msg = (
        'Please continue working on the task on whatever approach you think is suitable.\n'
        'If you think you have solved the task, please first send your answer to user through message and then finish the interaction.\n'
        f'{encaps_str}'
        'IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.\n'
    )

    if state.history:
        # check if the last action has an answer, if so, early exit
        if try_parse is not None:
            last_action = next(
                (
                    event
                    for event in reversed(state.history)
                    if isinstance(event, Action)
                ),
                None,
            )
            ans = try_parse(last_action)
            if ans is not None:
                return '/exit'

        # check if the agent has tried to talk to the user 3 times, if so, let the agent know it can give up
        user_msgs = [
            event
            for event in state.history
            if isinstance(event, MessageAction) and event.source == 'user'
        ]
        if len(user_msgs) >= 2:
            # let the agent know that it can give up when it has tried 3 times
            return (
                msg
                + 'If you want to give up, use the "finish" tool to finish the interaction.\n'
            )
    return msg


def cleanup():
    print('Cleaning up child processes...')
    for process in mp.active_children():
        print(f'Terminating child process: {process.name}')
        process.terminate()
        process.join()


def make_metadata(
    llm_config: LLMConfig,
    dataset_name: str,
    agent_class: str,
    max_iterations: int,
    eval_note: str | None,
    eval_output_dir: str,
    eval_workspace_base: str | None = None,
    data_split: str | None = None,
    details: dict[str, Any] | None = None,
    agent_config: AgentConfig | None = None,
    condenser_config: CondenserConfig | None = None,
) -> EvalMetadata:
    model_name = llm_config.model.split('/')[-1]
    model_path = model_name.replace(':', '_').replace('@', '-')
    eval_note = f'_N_{eval_note}' if eval_note else ''

    eval_output_path = os.path.join(
        eval_output_dir,
        dataset_name,
        agent_class,
        f'{model_path}_maxiter_{max_iterations}{eval_note}',
    )
    if eval_workspace_base:
        eval_workspace_path = os.path.join(
            eval_workspace_base,
            dataset_name,
            agent_class,
            f'{model_path}_maxiter_{max_iterations}{eval_note}',
        )
    else:
        eval_workspace_path = None

    pathlib.Path(eval_output_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(eval_output_path, 'logs')).mkdir(
        parents=True, exist_ok=True
    )
    logger.info(f'Using evaluation output directory: {eval_output_path}')
    
    if eval_workspace_path:
        pathlib.Path(eval_workspace_path).mkdir(parents=True, exist_ok=True)
        logger.info(f'Using evaluation workspace directory: {eval_workspace_path}')

    metadata = EvalMetadata(
        agent_class=agent_class,
        llm_config=llm_config,
        agent_config=agent_config,
        max_iterations=max_iterations,
        eval_output_dir=eval_output_path,
        eval_workspace_path=eval_workspace_path,
        start_time=time.strftime('%Y-%m-%d %H:%M:%S'),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'])
        .decode('utf-8')
        .strip(),
        dataset=dataset_name,
        data_split=data_split,
        details=details,
        condenser_config=condenser_config
        if condenser_config
        else NoOpCondenserConfig(),
    )
    metadata_json = metadata.model_dump_json()
    logger.info(f'Metadata: {metadata_json}')
    with open(os.path.join(eval_output_path, 'metadata.json'), 'w') as f:
        f.write(metadata_json)

    return metadata


def prepare_dataset(
    dataset: pd.DataFrame,
    output_file: str,
    eval_n_limit: int,
    eval_ids: list[str] | None = None,
    skip_num: int | None = None,
):
    assert (
        'instance_id' in dataset.columns
    ), "Expected 'instance_id' column in the dataset. You should define your own unique identifier for each instance and use it as the 'instance_id' column."
    id_column = 'instance_id'
    logger.info(f'Writing evaluation output to {output_file}')
    finished_ids: set[str] = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                finished_ids.add(str(data[id_column]))
        logger.warning(
            f'\nOutput file {output_file} already exists. Loaded {len(finished_ids)} finished instances.'
        )

    if eval_ids:
        eval_ids_converted = [dataset[id_column].dtype.type(id) for id in eval_ids]
        dataset = dataset[dataset[id_column].isin(eval_ids_converted)]
        logger.info(f'Limiting evaluation to {len(eval_ids)} specific instances.')
    elif skip_num and skip_num >= 0:
        skip_num = min(skip_num, len(dataset))
        dataset = dataset.iloc[skip_num:]
        logger.info(
            f'Starting evaluation with skipping first {skip_num} instances ({len(dataset)} instances to run).'
        )
        if eval_n_limit and eval_n_limit > 0:
            # Use fixed random seed 42 for sampling without replacement
            dataset = dataset.sample(
                min(eval_n_limit, len(dataset)), random_state=42, replace=False
            )
            logger.info(
                f'Randomly sampling {eval_n_limit} unique instances with random seed 42.'
            )
    elif eval_n_limit and eval_n_limit > 0:
        # Use fixed random seed 42 for sampling without replacement
        dataset = dataset.sample(
            min(eval_n_limit, len(dataset)), random_state=42, replace=False
        )
        logger.info(
            f'Randomly sampling {eval_n_limit} unique instances with random seed 42.'
        )

    new_dataset = [
        instance
        for _, instance in dataset.iterrows()
        if str(instance[id_column]) not in finished_ids
    ]
    logger.info(
        f'Finished instances: {len(finished_ids)}, Remaining instances: {len(new_dataset)}'
    )

    return pd.DataFrame(new_dataset)


def update_progress(
    result: EvalOutput,
    pbar: tqdm,
    output_fp: TextIO,
):
    """Update the progress bar and write the result to the output file."""
    pbar.update(1)
    pbar.set_description(f'Instance {result.instance_id}')
    pbar.set_postfix_str(f'Test Result: {str(result.test_result)[:300]}...')
    logger.info(
        f'Finished evaluation for instance {result.instance_id}: {str(result.test_result)[:300]}...\n'
    )
    output_fp.write(result.model_dump_json() + '\n')
    output_fp.flush()


def assert_and_raise(condition: bool, msg: str):
    """Raise an EvalException if the condition is not met.

    This will be used in conjunction with _process_instance_wrapper to handle retries. An EvalException should trigger a retry.
    """
    if not condition:
        raise EvalException(msg)


def _process_instance_wrapper(
    process_instance_func: Callable[[pd.Series, EvalMetadata, bool], EvalOutput],
    instance: pd.Series,
    metadata: EvalMetadata,
    use_mp: bool,
    max_retries: int = 5,
    timeout_seconds: int | None = None,
) -> EvalOutput:
    """Wrap the process_instance_func to handle retries and errors."""
    runtime_failure_count = 0
    for attempt in range(max_retries + 1):
        try:
            kwargs = {}
            # check if process_instance_func accepts timeout_seconds parameter
            sig = signature(process_instance_func)
            if 'runtime_failure_count' in sig.parameters:
                kwargs['runtime_failure_count'] = runtime_failure_count

            if timeout_seconds is not None:
                with timeout(timeout_seconds):
                    result = process_instance_func(instance, metadata, use_mp, **kwargs)
            else:
                result = process_instance_func(instance, metadata, use_mp, **kwargs)
            return result
        except EvalTimeoutException as e:
            error = f'Timeout after {timeout_seconds} seconds'
            stacktrace = traceback.format_exc()
            msg = (
                '-' * 10
                + '\n'
                + f'Timeout ({timeout_seconds} seconds) in instance [{instance.instance_id}], Stopped evaluation for this instance.'
                + '\n'
                + '-' * 10
            )
            logger.exception(e)
            return EvalOutput(
                instance_id=instance.instance_id,
                test_result={},
                error=error,
            )
        except Exception as e:
            error = str(e)
            stacktrace = traceback.format_exc()
            if attempt == max_retries:
                msg = (
                    '-' * 10
                    + '\n'
                    + f'Error in instance [{instance.instance_id}]: {error}. Stacktrace:\n{stacktrace}'
                    + '\n'
                    + f'[Encountered after {max_retries} retries. Please check the logs and report the issue.]'
                    + '-' * 10
                )
                # Raise an error after all retries & stop the evaluation
                logger.exception(e)
                raise RuntimeError(
                    f'Maximum error retries reached for instance {instance.instance_id}'
                ) from e
            msg = (
                '-' * 10
                + '\n'
                + f'Error in instance [{instance.instance_id}]: {error}. Stacktrace:\n{stacktrace}'
                + '\n'
                + '-' * 10
                + f'[The above error occurred. Retrying... (attempt {attempt + 1} of {max_retries})]'
                + '-' * 10
                + '\n'
            )
            # e is likely an EvalException, so we can't directly infer it from type
            # but rather check if it's a fatal error
            # But it can also be AgentRuntime**Error (e.g., swe_bench/eval_infer.py)
            _error_str = type(e).__name__ + ': ' + str(e)
            if is_fatal_runtime_error(_error_str):
                runtime_failure_count += 1
                msg += f'Runtime disconnected error detected for instance {instance.instance_id}, runtime failure count: {runtime_failure_count}'
                msg += '\n' + '-' * 10 + '\n'
            logger.error(msg)
            time.sleep(5)


def _process_instance_wrapper_mp(args):
    """Wrapper for multiprocessing, especially for imap_unordered."""
    return _process_instance_wrapper(*args)


def run_evaluation(
    dataset: pd.DataFrame,
    metadata: EvalMetadata | None,
    output_file: str,
    num_workers: int,
    process_instance_func: Callable[
        [pd.Series, EvalMetadata, bool], Awaitable[EvalOutput]
    ],
    max_retries: int = 5,  # number of retries for each instance
    timeout_seconds: int | None = None,
):
    use_multiprocessing = num_workers > 1

    if metadata is not None:
        logger.info(
            f'Evaluation started with Agent {metadata.agent_class}:\n'
            f'model {metadata.llm_config.model}, max iterations {metadata.max_iterations}.\n'
        )
    else:
        logger.warning('Running evaluation without metadata.')
        logger.info(f'Evaluation started with {num_workers} workers.')

    total_instances = len(dataset)
    pbar = tqdm(total=total_instances, desc='Instances processed')
    output_fp = open(output_file, 'a')

    try:
        if use_multiprocessing:
            with mp.Pool(num_workers) as pool:
                args_iter = (
                    (
                        process_instance_func,
                        instance,
                        metadata,
                        True,
                        max_retries,
                        timeout_seconds,
                    )
                    for _, instance in dataset.iterrows()
                )
                results = pool.imap_unordered(_process_instance_wrapper_mp, args_iter)
                for result in results:
                    update_progress(result, pbar, output_fp)
        else:
            for _, instance in dataset.iterrows():
                result = _process_instance_wrapper(
                    process_instance_func=process_instance_func,
                    instance=instance,
                    metadata=metadata,
                    use_mp=False,
                    max_retries=max_retries,
                )
                update_progress(result, pbar, output_fp)

    except KeyboardInterrupt:
        print('\nKeyboardInterrupt received. Cleaning up...\n')
        cleanup()

    output_fp.close()
    logger.info('\nEvaluation finished.\n')


def reset_logger_for_multiprocessing(
    logger: logging.Logger, instance_id: str, log_dir: str
):
    """Reset the logger for multiprocessing.

    Save logs to a separate file for each process, instead of trying to write to the
    same file/console from multiple processes.
    """
    # Set up logger
    log_file = os.path.join(
        log_dir,
        f'instance_{instance_id}.log',
    )
    # Remove all existing handlers from logger
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # add console handler to print ONE line
    console_handler = get_console_handler(log_level=logging.INFO)
    console_handler.setFormatter(
        logging.Formatter(
            f'Instance {instance_id} - ' + '%(asctime)s - %(levelname)s - %(message)s'
        )
    )
    logger.addHandler(console_handler)
    logger.info(
        f'Starting evaluation for instance {instance_id}.\n'
        f'Hint: run "tail -f {log_file}" to see live logs in a separate shell'
    )
    # Only log WARNING or higher to console
    console_handler.setLevel(logging.WARNING)

    # Log INFO and above to file
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)


def update_llm_config_for_completions_logging(
    llm_config: LLMConfig,
    eval_output_dir: str,
    instance_id: str,
) -> LLMConfig:
    """Update the LLM config for logging completions."""
    if llm_config.log_completions:
        llm_config.log_completions_folder = os.path.join(
            eval_output_dir, 'llm_completions', instance_id
        )
        logger.info(
            f'Logging LLM completions for instance {instance_id} to '
            f'{llm_config.log_completions_folder}'
        )
    return llm_config


# history is now available as a filtered stream of events, rather than list of pairs of (Action, Observation)
# we rebuild the pairs here
# for compatibility with the existing output format in evaluations
# remove this when it's no longer necessary
def compatibility_for_eval_history_pairs(
    history: list[Event],
) -> list[tuple[dict, dict]]:
    history_pairs = []

    for action, observation in get_pairs_from_events(history):
        history_pairs.append((event_to_dict(action), event_to_dict(observation)))

    return history_pairs


def is_fatal_evaluation_error(error: str | None) -> bool:
    if not error:
        return False

    FATAL_EXCEPTIONS = [
        AgentRuntimeError,
        AgentRuntimeBuildError,
        AgentRuntimeTimeoutError,
        AgentRuntimeUnavailableError,
        AgentRuntimeNotReadyError,
        AgentRuntimeDisconnectedError,
        AgentRuntimeNotFoundError,
        ConnectionError,
    ]

    if any(exception.__name__ in error for exception in FATAL_EXCEPTIONS):
        logger.error(f'Fatal evaluation error detected: {error}')
        return True

    return False


def is_fatal_runtime_error(error: str | None) -> bool:
    if not error:
        return False

    FATAL_RUNTIME_ERRORS = [
        AgentRuntimeTimeoutError,
        AgentRuntimeUnavailableError,
        AgentRuntimeDisconnectedError,
        AgentRuntimeNotFoundError,
    ]

    if any(exception.__name__ in error for exception in FATAL_RUNTIME_ERRORS):
        logger.error(f'Fatal runtime error detected: {error}')
        return True

    return False


def get_metrics(state: State) -> dict[str, Any]:
    """Extract metrics from the state."""
    metrics = state.metrics.get() if state.metrics else {}
    metrics['condenser'] = get_condensation_metadata(state)
    return metrics


def get_default_sandbox_config_for_eval() -> SandboxConfig:
    return SandboxConfig(
        use_host_network=False,
        # large enough timeout, since some testcases take very long to run
        timeout=300,
        api_key=os.environ.get('ALLHANDS_API_KEY', None),
        remote_runtime_api_url=os.environ.get('SANDBOX_REMOTE_RUNTIME_API_URL'),
        keep_runtime_alive=False,
        remote_runtime_init_timeout=3600,
        remote_runtime_api_timeout=120,
        remote_runtime_enable_retries=True,
        remote_runtime_class='sysbox',
        docker_runtime_kwargs={
            'userns_mode': 'host'
        }
    )
