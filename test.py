from code_executor import PythonExecutor
import multiprocess

if __name__ == '__main__':
    multiprocess.set_start_method('spawn')

    current_code = """
```python
def calculate_hydrogen_mass(mass_of_water_grams):
    mass_of_hydrogen = 1.00794  # g/mol
    mass_of_water = 18.01528  # g/mol
    ratio = (2 * mass_of_hydrogen) / mass_of_water
    return ratio * mass_of_water_grams

mass_of_water = 23.5  # grams
hydrogen_mass = calculate_hydrogen_mass(mass_of_water)

print(hydrogen_mass)
```
    """
    executor = PythonExecutor(get_answer_from_stdout=True)
    result, report = executor.apply(current_code)
    print("Result:", result)
    print("Report:", report)

    # Make sure to close the pool when done
    executor.pool.close()
    executor.pool.join()
