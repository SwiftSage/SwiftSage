"""
Source and credits: https://github.com/ZubinGou/math-evaluation-harness/blob/main/python_executor.py

We modified it to be more simple.
"""

import io
import pickle
import traceback
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import multiprocessing
import time
from contextlib import redirect_stdout


class GenericRuntime:
    GLOBAL_DICT = {}
    LOCAL_DICT = None
    HEADERS = []

    def __init__(self):
        self._global_vars = self.GLOBAL_DICT.copy()
        self._local_vars = self.LOCAL_DICT.copy() if self.LOCAL_DICT else None

        for c in self.HEADERS:
            self.exec_code(c)

    def exec_code(self, code_piece: str) -> None:
        exec(code_piece, self._global_vars)

    def eval_code(self, expr: str) -> any:
        return eval(expr, self._global_vars)

    def inject(self, var_dict):
        self._global_vars.update(var_dict)

    @property
    def answer(self):
        return self._global_vars['answer']


class PythonExecutor:
    def __init__(
        self,
        runtime=None,
        get_answer_symbol=None,
        get_answer_expr=None,
        get_answer_from_stdout=False,
        timeout=10,
    ):
        self.runtime = runtime if runtime else GenericRuntime()
        self.answer_symbol = get_answer_symbol
        self.get_answer_expr = get_answer_expr
        self.get_answer_from_stdout = get_answer_from_stdout
        self.timeout = timeout

    def _run_code(self, code, result_queue):
        try:
            if self.get_answer_from_stdout:
                program_io = io.StringIO()
                with redirect_stdout(program_io):
                    self.runtime.exec_code('\n'.join(code))
                program_io.seek(0)
                result = program_io.read()
            elif self.answer_symbol:
                self.runtime.exec_code('\n'.join(code))
                result = self.runtime._global_vars[self.answer_symbol]
            elif self.get_answer_expr:
                self.runtime.exec_code('\n'.join(code))
                result = self.runtime.eval_code(self.get_answer_expr)
            else:
                self.runtime.exec_code('\n'.join(code[:-1]))
                result = self.runtime.eval_code(code[-1])

            report = "Done"
            pickle.dumps(result)  # Serialization check
            result_queue.put((result, report))  # Send result and report back to main process

        except Exception as e:
            result_queue.put(('', str(e)))

    def apply(self, code):
        code_snippet = code.split('\n')
        # code_snippet = code
        try:
            # Start a separate process to run the code
            result_queue = multiprocessing.Queue()
            process = multiprocessing.Process(target=self._run_code, args=(code_snippet, result_queue))
            process.start()
            process.join(timeout=self.timeout)

            if process.is_alive():
                process.terminate()  # Terminate the process if it's still running
                result = ''
                report = "Timeout Error"
            else:
                result, report = result_queue.get()
                result = result.strip()

        except Exception as e:
            result = ''
            report = str(e)

        return result, report


# Example usage
if __name__ == "__main__":
    executor = PythonExecutor(get_answer_from_stdout=True)
    code = """
import math\n\n# Define the reduced Planck constant\nh_bar = 6.582 * (10**-16)  # eV sec\n\n# Define the lifetimes of the quantum states\nlifetime1 = 10**-9  # sec\nlifetime2 = 10**-8  # sec\n\n# Calculate the minimum energy difference using the Heisenberg uncertainty principle\nmin_energy_diff = h_bar / max(lifetime1, lifetime2)\n\n# Print the minimum energy difference\nprint(\"The minimum energy difference required to clearly distinguish the two energy levels is:\", min_energy_diff, \"eV\")\n\n# Check which option is the right choice\nif min_energy_diff >= 10**-4:\n    print(\"The right choice is (C) 10^-4 eV\")\nelif min_energy_diff >= 10**-11:\n    print(\"The right choice is (D) 10^-11 eV\")\nelse:\n    print(\"The right choice is (A) 10^-9 eV or (B) 10^-8 eV\")
"""
    result, report = executor.apply(code)
    print("Result:", result)
    print("Report:", report)