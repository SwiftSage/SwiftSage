"""
Source and credits: https://github.com/ZubinGou/math-evaluation-harness/blob/main/python_executor.py

We modified it to be more simple.
"""

import io
import pickle
import traceback
from concurrent.futures import ProcessPoolExecutor, TimeoutError
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
        timeout_length=15,
    ):
        self.runtime = runtime if runtime else GenericRuntime()
        self.answer_symbol = get_answer_symbol
        self.get_answer_expr = get_answer_expr
        self.get_answer_from_stdout = get_answer_from_stdout
        self.timeout_length = timeout_length

    def execute(self, code):
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
        except Exception as e:
            result = ''
            report = str(e)

        return result, report

    def apply(self, code):
        code_snippet = code.split('\n')

        # Use ProcessPoolExecutor to enforce timeout
        with ProcessPoolExecutor() as executor:
            future = executor.submit(self.execute, code_snippet)
            try:
                result, report = future.result(timeout=self.timeout_length)
            except TimeoutError:
                result, report = "", "Timeout Error"

        return result.strip(), report.strip()


# Example usage
if __name__ == "__main__":
    executor = PythonExecutor(get_answer_from_stdout=True)
    code = """
from sympy import Matrix

def null_space_basis():
    A = Matrix([[3, 3, -1, -6], [9, -1, -8, -1], [7, 4, -2, -9]])
    basis = A.nullspace()
    return [v.evalf(3) for v in basis]

result = null_space_basis()
print(result)
"""
    result, report = executor.apply(code)
    print("Result:", result)
    print("Report:", report)