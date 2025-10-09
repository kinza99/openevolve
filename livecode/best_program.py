from typing import Callable
import concurrent.futures
import time
from tqdm import tqdm

BATCH_SIZE = 32 
BATCH_DELAY = 1
PARALLEL_WORKERS = 256
def verify_parallel(tasks : list[tuple[dict, list[str], list[str], Callable]]):
    
    # Build unique sets of solution_ids and testing_ids
    solution_ids = set()
    testing_ids = set()
    for task in tasks:
        _, solution_id, testing_id, _ = task
        solution_ids.add(solution_id)
        testing_ids.add(testing_id)
    solution_pass_testing = {solution_id: [] for solution_id in solution_ids}
    testing_accept_solutions = {testing_id: [] for testing_id in testing_ids}
        
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
        for future in concurrent.futures.as_completed(all_futures):

            result = future.result()
            question_title = result['question_title']
            solution_id = result['solution_id']
            testing_id = result['testing_id']
            if result['result'] and not result['unknown_error']:
                solution_pass_testing[solution_id].append(testing_id)
                testing_accept_solutions[testing_id].append(solution_id)
    
    return solution_pass_testing, testing_accept_solutions
"""
Given a instance, which contains the os interaction task instructions, the function is choosing the best solution to solve the task and the best testing to test the solution.
instance: dict, which contains the os interaction task instructions
solution_id_list: list of solution ids, which are the ids of the solutions to the task
testing_id_list: list of testing ids, which are the ids of the testings to the solutions
verify_func: function to verify if the solution can pass the testing
return: tuple, which contains the best solution id and the best testing id
"""

# EVOLVE-BLOCK-START
def strategy(instance : dict, solution_id_list : list[str], testing_id_list : list[str], verify_func : Callable):
   
    # Initialize the testing_accept_solutions dictionary with empty lists for each testing_id
    testing_accept_solutions = {testing_id: [] for testing_id in testing_id_list}
    
    tasks = []
    for solution_id in solution_id_list:
        for testing_id in testing_id_list:
            tasks.append((instance, solution_id, testing_id, verify_func))
    solution_pass_testing, testing_accept_solutions = verify_parallel(tasks)
    
    # Compute solution_quality: fraction of testings passed by the solution.
    total_testings = len(testing_id_list)
    solution_quality = {}
    for solution_id in solution_id_list:
        count = len(solution_pass_testing[solution_id])
        solution_quality[solution_id] = count / total_testings

    # Compute testing_quality: difference between average solution_quality of passed vs failed solutions
    testing_quality = {}
    total_quality_all = sum(solution_quality.values())
    total_solutions = len(solution_id_list)
    for testing_id in testing_id_list:
        passed_solutions = testing_accept_solutions[testing_id]
        n_passed = len(passed_solutions)
        total_quality_passed = 0.0
        if n_passed > 0:
            for sol_id in passed_solutions:
                total_quality_passed += solution_quality[sol_id]
            passed_avg = total_quality_passed / n_passed
        else:
            passed_avg = 0.0

        n_failed = total_solutions - n_passed
        if n_failed > 0:
            total_quality_failed = total_quality_all - total_quality_passed
            failed_avg = total_quality_failed / n_failed
        else:
            failed_avg = 0.0

        testing_quality[testing_id] = passed_avg - failed_avg

    # Sort solutions by solution_quality (descending) and testings by testing_quality (descending)
    sorted_solutions = sorted(solution_id_list, key=lambda x: solution_quality[x], reverse=True)
    sorted_testings = sorted(testing_id_list, key=lambda x: testing_quality[x], reverse=True)
        
    return sorted_solutions, sorted_testings

# EVOLVE-BLOCK-END