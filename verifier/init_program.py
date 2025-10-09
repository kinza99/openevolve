import os
from typing import Callable
import concurrent.futures
import time
from tqdm import tqdm
from shared import (
    delete_container,
    get_runtime_id,
)
from openhands.core.logger import openhands_logger as logger
import traceback

BATCH_SIZE = 32 
BATCH_DELAY = 5 
PARALLEL_WORKERS = 128
def verify_parallel(tasks : list[tuple[dict, list[str], list[str], Callable]]):
    
    solution_pass_testing = {}
    testing_accept_solutions = {}
    
    for instance, solution_id, testing_id, verify_func in tasks:
        solution_pass_testing[solution_id] = []
        testing_accept_solutions[testing_id] = []
        
    all_futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        for batch_start in range(0, len(tasks), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(tasks))
            batch_tasks = tasks[batch_start:batch_end]
            for task in batch_tasks:
                instance, solution_id, testing_id, verify_func = task
                all_futures.append(executor.submit(verify_func, instance, solution_id, testing_id))
            if batch_end < len(tasks):
                time.sleep(BATCH_DELAY)
        for future in tqdm(concurrent.futures.as_completed(all_futures), desc='Strategy', total=len(all_futures), dynamic_ncols=True):

            result = future.result()
            instance_id = result['instance_id']
            solution_id = result['solution_id']
            testing_id = result['testing_id']
            if result['test_result']['result'] and result['test_result']['agent_answer'] != 'UNKNOWN_ERROR':
                solution_pass_testing[solution_id].append(testing_id)
                testing_accept_solutions[testing_id].append(solution_id)

            try:
                delete_container(get_runtime_id(instance_id, solution_id, testing_id))
            except Exception as e:
                logger.error(f'Failed to delete container: {e}')
    
    return solution_pass_testing, testing_accept_solutions


# EVOLVE-BLOCK-START
def strategy(instance : dict, solution_id_list : list[str], testing_id_list : list[str], verify_func : Callable):
    """
    Given a instance, which contains the os interaction task instructions, the function is choosing the best solution to solve the task and the best testing to test the solution.
    instance: dict, which contains the os interaction task instructions
    solution_id_list: list of solution ids, which are the ids of the solutions to the task
    testing_id_list: list of testing ids, which are the ids of the testings to the solutions
    verify_func: function to verify if the solution can pass the testing
    return: tuple, which contains the best solution id and the best testing id
    """
    solution_weights = {}
    testing_weights = {}
    testing_accept_solutions = {}
    for solution_id in solution_id_list:
        solution_weights[solution_id] = 0

    for testing_id in testing_id_list:
        testing_weights[testing_id] = 0
        testing_accept_solutions[testing_id] = []
    
    tasks = []
    for solution_id in solution_id_list:
        for testing_id in testing_id_list:
            tasks.append((instance, solution_id, testing_id, verify_func))
    solution_pass_testing, testing_accept_solutions = verify_parallel(tasks)
    
    for solution_id in solution_id_list:
        solution_weights[solution_id] = len(solution_pass_testing[solution_id])
    for testing_id in testing_id_list:
        testing_weights[testing_id] = len(testing_accept_solutions[testing_id])
        
    return sorted(solution_id_list, key=lambda x: solution_weights[x], reverse=True), sorted(testing_id_list, key=lambda x: testing_weights[x], reverse=True)

# EVOLVE-BLOCK-END