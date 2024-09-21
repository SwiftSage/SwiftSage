from code_executor import PythonExecutor
import multiprocess

if __name__ == '__main__':
    multiprocess.set_start_method('spawn')

    current_code = """
from collections import Counter

def count_letter_r(word):
    word = word.lower()
    return Counter(word)['r']

result = count_letter_r("strawberry")
print(f"There are {result} letter r in the word 'strawberry'.")
    """
    executor = PythonExecutor(get_answer_from_stdout=True)
    result, report = executor.apply(current_code)
    print("Result:", result)
    print("Report:", report)

    # Make sure to close the pool when done
    executor.pool.close()
    executor.pool.join()
