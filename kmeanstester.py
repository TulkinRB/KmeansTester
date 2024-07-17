import random
import re
import string
import subprocess
import enum
import sys

from sklearn.cluster import KMeans
from sklearn.cluster import _kmeans

ERROR_MESSAGE = "An Error Has Occurred\n"
K_ERROR_MESSAGE = "Invalid number of clusters!\n"
ITER_ERROR_MESSAGE = "Invalid maximum iteration!\n"

VECTORS_FILE_PATH = "kmeans_tester/vectors.txt"
PARAMETERS_FILE_PATH = "kmeans_tester/params.txt"
EXPECTED_OUTPUT_FILE_PATH = "kmeans_tester/expected_output.txt"
RECEIVED_OUTPUT_FILE_PATH = "kmeans_tester/received_output.txt"
INSTRUCTION_FILE_PATH = "kmeans_tester/instruction.txt"
ALLOC_CALLS_FILE_PATH = "kmeans_tester/alloc_calls.txt"

SHORT_TEST_SIZE = {
    "NUM_OF_INVALID_TESTS": 30,
    "NUM_OF_REGULAR_TESTS": 5,
    "NUM_OF_REGULAR_FULL_TESTS": 1,
    "NUM_OF_LARGE_TESTS": 1,
    "NUM_OF_LARGE_FULL_TESTS": 1,
}
LONG_TEST_SIZE = {
    "NUM_OF_INVALID_TESTS": 300,
    "NUM_OF_REGULAR_TESTS": 20,
    "NUM_OF_REGULAR_FULL_TESTS": 5,
    "NUM_OF_LARGE_TESTS": 5,
    "NUM_OF_LARGE_FULL_TESTS": 2,
}
MAX_COORD_VALUE = 1000
NUM_OF_FAILED_ALLOC_CHECKS_PER_TEST = 30

valgrind_leak_regex = re.compile("All heap blocks were freed -- no leaks are possible")

_kmeans._tolerance = lambda X, tol: tol  # This stupid workaround is stupid


class InputValidity(enum.Flag):
    K_FORMAT = enum.auto()
    K_RANGE = enum.auto()
    ITER = enum.auto()
    INVALID_LENGTH = enum.auto()
    K = K_FORMAT | K_RANGE
    VALID = K | ITER
    PRE_VECTORS_VALID = K_FORMAT | ITER


class TestFailed(Exception):
    
    def __init__(self, case):
        super().__init__()
        self.case = case


def generate_vector(length):
    return (round(random.uniform(-MAX_COORD_VALUE, MAX_COORD_VALUE), 4) for i in range(length))


def generate_vectors(length, count):
    for i in range(count):
        yield generate_vector(length)


def vectors_as_lists(vectors_gen):
    return [list(vector_gen) for vector_gen in vectors_gen]


def format_vectors(vectors, debug=False):
    return "\n".join(
        ",".join("{:.4f}".format(coord) for coord in vector)
        # + ("" if not debug else ("\n" + ",".join("{}".format(coord) for coord in vector)))
        for vector in vectors
    ) + "\n\n"


def write_vectors(vectors, filename):
    with open(filename, "w") as file:
        file.write(format_vectors(vectors))


def benchmark_kmeans(k, num_iter, vectors):
    if num_iter is None:
        num_iter = 200
    kmeans = KMeans(n_clusters=k, max_iter=num_iter, n_init=1, tol=0.001, init=vectors[:k])
    kmeans.fit(vectors)
    return [
        [float(coord) for coord in centroid]
        for centroid in kmeans.cluster_centers_
    ]


class ProgressIndicator:

    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'

    MAIN_BAR_LENGTH = 50
    SECONDARY_BAR_LENGTH = 20

    def __init__(self, num_tests):
        self.num_tests = num_tests
        self.curr_test = 0
        self.py = False
        self.c = False
        self.c_alloc_fail = (0, 0)
        self.valgrind = False
        self.valgrind_alloc_fail = (0, 0)
        self.printed = False

    def reset_inner_progress(self):
        if self.py is not None:
            self.py = False
        self.c = False
        self.c_alloc_fail = (0, 0)
        self.valgrind = False
        self.valgrind_alloc_fail = (0, 0)

    def refresh(self):
        output = ""
        if self.printed:
            output += 7 * self.LINE_UP
        if self.py is not None:
            output += self.LINE_UP
        for line in self.generate():
            output += self.LINE_CLEAR
            output += line

        print(output, flush=True)
        self.printed = True

    def generate(self):
        lines = [
            self.generate_bar_string(self.MAIN_BAR_LENGTH, self.curr_test, self.num_tests) + " test cases completed\n",
            "\n",
            "C code test: " + ("Completed" if self.c else "") + "\n",
            "C code test with forced allocation fails: " + self.generate_bar_string(self.SECONDARY_BAR_LENGTH, *self.c_alloc_fail) + " completed\n",
            "C code test with valgrind: " + ("Completed" if self.valgrind else "") + "\n",
            "C code test with valgrind and forced allocation fails: " + self.generate_bar_string(self.SECONDARY_BAR_LENGTH, *self.valgrind_alloc_fail) + " completed\n",
        ]
        if self.py is not None:
            lines.insert(2, "Python code test: " + ("Completed" if self.py else "") + "\n")
        return lines

    @classmethod
    def generate_bar_string(cls, size, completed, total):
        if total == 0:
            full_length = 0
        else:
            full_length = int(size * completed / total)
        return (
            "["
            + "=" * full_length
            + "." * (size - full_length)
            + "]    "
            + f"{completed}/{total}"
        )


class TestCase:

    def __init__(self, k, num_iter, vectors, input_validity, full, python, params=None):
        self.k = k
        self.num_iter = num_iter
        self.vectors = vectors
        self.input_validity = input_validity
        if input_validity is InputValidity.VALID:
            k = int(float(k))
            num_iter = None if num_iter is None else int(float(num_iter))
            self.expected_output = benchmark_kmeans(k, num_iter, vectors)
        self.full = full
        self.python = python
        self.params = params

    def validate_output(self, test_type, output, alloc_fail=False):
        expected_outputs = list()

        if self.input_validity is InputValidity.VALID:
            if not alloc_fail:
                return self.validate_vector_output(test_type, output)
            expected_outputs.append(ERROR_MESSAGE)
        else:
            if self.input_validity is InputValidity.INVALID_LENGTH:
                expected_outputs.append(ERROR_MESSAGE)
            elif self.input_validity is InputValidity.PRE_VECTORS_VALID:
                if alloc_fail:
                    expected_outputs.append(ERROR_MESSAGE)
                else:
                    expected_outputs.append(K_ERROR_MESSAGE)
            else:
                if InputValidity.K not in self.input_validity:
                    expected_outputs.append(K_ERROR_MESSAGE)
                if InputValidity.ITER not in self.input_validity:
                    expected_outputs.append(ITER_ERROR_MESSAGE)

        if output not in expected_outputs:
            if output + "\n" in expected_outputs:
                print(f"\n\n{test_type}: Missing newline at the end of the output.")
                raise TestFailed(self)
            with open(RECEIVED_OUTPUT_FILE_PATH, "w") as f:
                f.write(output)
            print(
                f"\n\n{test_type}: Unexpected output. Expected one of the outputs {expected_outputs}. \n"
                f"Received output saved to {RECEIVED_OUTPUT_FILE_PATH}."
            )

            raise TestFailed(self)

    def validate_vector_output(self, test_type, output):
        output_lines = output.split("\n")
        fail = False
        if len(output_lines) not in (len(self.expected_output) + 1, len(self.expected_output) + 2):
            fail = True
        try:
            for expected_vector, line in zip(self.expected_output, output_lines):
                line = line.split(",")
                if len(line) != len(expected_vector):
                    fail = True
                    break
                for expected_coord, received_coord in zip(expected_vector, line):
                    coord_parts = received_coord.split(".")
                    if len(coord_parts) != 2 or len(coord_parts[1]) != 4:
                        fail = True
                        break
                    if abs(expected_coord - float(received_coord)) > 0.0001:  # floating point errors are so fun!
                        fail = True
                        break
        except ValueError:
            fail = True
        if output_lines[-1] != "":
            fail = True
        if len(output_lines) == len(self.expected_output) + 2 and output_lines[-2] != "":
            fail = True

        if fail:
            with open(RECEIVED_OUTPUT_FILE_PATH, "w") as f:
                f.write(output)
            with open(EXPECTED_OUTPUT_FILE_PATH, "w") as f:
                f.write(format_vectors(self.expected_output))
            print(
                f"\n\n{test_type}: Unexpected output. Expected output saved to {EXPECTED_OUTPUT_FILE_PATH}. \n"
                f"Received output saved to {RECEIVED_OUTPUT_FILE_PATH}."
            )
            raise TestFailed(self)

        if len(output_lines) != len(self.expected_output) + 2:
            print(f"\n\n{test_type}: Missing newline at the end of the output.")
            raise TestFailed(self)

    def validate_valgrind_output(self, test_type, output):
        if valgrind_leak_regex.search(output) is None:
            print(f"\n\n{test_type}: Memory leak detected. Valgrind output:\n\n{output}")
            raise TestFailed(self)

    def test_all(self, progress):
        if self.python:
            self.python_test()
            progress.py = True
            progress.refresh()
        self.basic_c_test()
        progress.c = True
        progress.refresh()
        self.basic_alloc_fail_c_test(progress)
        self.valrgrind_c_test()
        progress.valgrind = True
        progress.refresh()
        self.valgrind_alloc_fail_c_test(progress)

    def python_test(self):
        write_vectors(self.vectors, VECTORS_FILE_PATH)
        if self.params is not None:
            cmd = ["python", "kmeans.py"] + self.params + [VECTORS_FILE_PATH]
        else:
            cmd = ["python", "kmeans.py", str(self.k), VECTORS_FILE_PATH]
            if self.num_iter is not None:
                cmd.insert(3, str(self.num_iter))

        sp = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        self.validate_output("Python Test", sp.stdout)

    def basic_c_test(self):
        if self.params is not None:
            cmd = ["python", "kmeans.py"] + self.params
        else:
            cmd = ["./kmeans", str(self.k)]
            if self.num_iter is not None:
                cmd += [str(self.num_iter)]

        with open(INSTRUCTION_FILE_PATH, "w") as f:
            f.write("-1")
        with open(ALLOC_CALLS_FILE_PATH, "w") as f:
            ...

        sp = subprocess.run(cmd, stdout=subprocess.PIPE, input=format_vectors(self.vectors), text=True)
        if self.input_validity is InputValidity.VALID and sp.returncode != 0:
            print(f"\n\nBasic C Test: Unexpected return code: {sp.returncode}. Expected 0.")
            raise TestFailed(self)
        if self.input_validity is not InputValidity.VALID and sp.returncode != 1:
            print(f"\n\nBasic C Test: Unexpected return code: {sp.returncode}. Expected 1.")
            raise TestFailed(self)
        self.validate_output("Basic C Test", sp.stdout)

    def basic_alloc_fail_c_test(self, progress):
        if self.params is not None:
            cmd = ["python", "kmeans.py"] + self.params
        else:
            cmd = ["./kmeans", str(self.k)]
            if self.num_iter is not None:
                cmd += [str(self.num_iter)]

        with open(ALLOC_CALLS_FILE_PATH, "r") as f:
            num_of_calls = len(f.read())

        if self.full:
            failed_allocs = range(num_of_calls)
            num_of_checks = num_of_calls
        else:
            num_of_checks = min(NUM_OF_FAILED_ALLOC_CHECKS_PER_TEST, num_of_calls)
            failed_allocs = random.sample(range(num_of_calls), num_of_checks)
        for i, alloc in enumerate(failed_allocs):
            progress.c_alloc_fail = (i, num_of_checks)
            progress.refresh()
            with open(INSTRUCTION_FILE_PATH, "w") as f:
                f.write(str(alloc))
            sp = subprocess.run(cmd, stdout=subprocess.PIPE, input=format_vectors(self.vectors), text=True)
            if sp.returncode != 1:
                print(f"\n\nBasic Alloc Fail C Test: Unexpected return code: {sp.returncode}. Expected 1.")
                raise TestFailed(self)
            self.validate_output("Basic Alloc Fail C Test", str(sp.stdout), alloc_fail=True)
        progress.c_alloc_fail = (num_of_checks, num_of_checks)
        progress.refresh()

    def valrgrind_c_test(self):
        if self.params is not None:
            cmd = ["python", "kmeans.py"] + self.params
        else:
            cmd = ["valgrind", "./kmeans", str(self.k)]
            if self.num_iter is not None:
                cmd += [str(self.num_iter)]

        with open(INSTRUCTION_FILE_PATH, "w") as f:
            f.write("-1")
        with open(ALLOC_CALLS_FILE_PATH, "w") as f:
            ...

        sp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, input=format_vectors(self.vectors), text=True)
        self.validate_output("Valgrind C Test", sp.stdout)
        self.validate_valgrind_output("Valgrind C Test", sp.stderr)

    def valgrind_alloc_fail_c_test(self, progress):
        if self.params is not None:
            cmd = ["python", "kmeans.py"] + self.params
        else:
            cmd = ["valgrind", "./kmeans", str(self.k)]
            if self.num_iter is not None:
                cmd += [str(self.num_iter)]

        with open(ALLOC_CALLS_FILE_PATH, "r") as f:
            num_of_calls = len(f.read())

        if self.full:
            failed_allocs = range(num_of_calls)
            num_of_checks = num_of_calls
        else:
            num_of_checks = min(NUM_OF_FAILED_ALLOC_CHECKS_PER_TEST, num_of_calls)
            failed_allocs = random.sample(range(num_of_calls), num_of_checks)
        for i, alloc in enumerate(failed_allocs):
            progress.valgrind_alloc_fail = (i, num_of_checks)
            progress.refresh()
            with open(INSTRUCTION_FILE_PATH, "w") as f:
                f.write(str(alloc))
            sp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, input=format_vectors(self.vectors), text=True)
            self.validate_output("Valgrind Alloc Fail C Test", str(sp.stdout), alloc_fail=True)
            self.validate_valgrind_output("Valgrind Alloc Fail C Test", sp.stderr)
        progress.valgrind_alloc_fail = (num_of_checks, num_of_checks)
        progress.refresh()


def generate_regular_case(full=False, python=True):
    vector_length = random.randint(2, 10)
    vector_count = random.randint(3, 1000)
    k = random.randint(2, vector_count - 1)
    if random.randint(0, 4) <= 1:
        k = str(k) + "." + "0" * random.randint(1, 4)
    num_iter = None
    if random.randint(0, 10) >= 3:
        num_iter = random.randint(2, 999)
        if random.randint(0, 4) <= 1:
            num_iter = str(num_iter) + "." + "0" * random.randint(1, 4)
    vectors = vectors_as_lists(generate_vectors(vector_length, vector_count))
    return TestCase(k, num_iter, vectors, InputValidity.VALID, full, python)


def generate_large_case(full=False, python=False):
    vector_length = random.randint(11, 512)
    vector_count = random.randint(1001, 16384)
    k = random.randint(2, vector_count - 1)
    num_iter = None
    if random.randint(0, 10) >= 3:
        num_iter = random.randint(2, 150)
    vectors = vectors_as_lists(generate_vectors(vector_length, vector_count))
    return TestCase(k, num_iter, vectors, InputValidity.VALID, full, python)


def generate_invalid_case(full=False, python=True):
    vector_length = random.randint(2, 10)
    vector_count = random.randint(3, 1000)
    vectors = vectors_as_lists(generate_vectors(vector_length, vector_count))
    mode = random.randint(0, 3)
    if mode == 4:
        param_length = random.randint(0, 8)
        if param_length != 0:
            param_length += 2
        params = [
            str(random.randint(1, 1000))
            for i in range(param_length)
        ]
        return TestCase(None, None, vectors, InputValidity.INVALID_LENGTH, full, python, params=params)
    validity = InputValidity(0)
    if mode == 0:
        validity |= InputValidity.K
        k = random.randint(2, vector_count - 1)
    else:
        if random.randint(0, 2) == 0:
            k = random.randint(0, 10) + vector_count
            validity |= InputValidity.K_FORMAT
        elif random.randint(0, 1) == 0:
            k = random.randint(-10, 1)
        else:
            k = "".join(random.choices(string.ascii_letters + string.punctuation, k=random.randint(1, 10)))

    if mode == 1:
        validity |= InputValidity.ITER
        num_iter = None
        if random.randint(0, 10) >= 3:
            num_iter = random.randint(2, 999)
    else:
        if random.randint(0, 2) == 0:
            num_iter = random.randint(1000, 1010)
        elif random.randint(0, 1) == 0:
            num_iter = random.randint(-10, 1)
        else:
            num_iter = "".join(random.choices(string.ascii_letters + string.punctuation, k=random.randint(1, 10)))

    return TestCase(k, num_iter, vectors, validity, full, python)


def perform_tests(num, generate_test_case, full=False, python=False):
    progress = ProgressIndicator(num)
    if not python:
        progress.py = None
    for i in range(num):
        progress.reset_inner_progress()
        progress.curr_test = i
        progress.refresh()
        generate_test_case(full, python).test_all(progress)
    progress.curr_test = num
    progress.refresh()


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "long":
        size = LONG_TEST_SIZE
        print("Performing a long test:\n")
    else:
        size = SHORT_TEST_SIZE
        print("Performing a short test:\n")
    try:
        print(f"\n\nPerforming {size['NUM_OF_INVALID_TESTS']} invalid input tests on Python and C code (Step 1 of 5):\n")
        perform_tests(size['NUM_OF_INVALID_TESTS'], generate_invalid_case, False, True)

        print(f"\n\nPerforming {size['NUM_OF_REGULAR_TESTS']} valid input tests on Python and C code (Step 2 of 5):\n")
        perform_tests(size['NUM_OF_REGULAR_TESTS'], generate_regular_case, False, True)

        print(f"\n\nPerforming {size['NUM_OF_REGULAR_FULL_TESTS']} full valid input tests on C code (Step 3 of 5):\n")
        perform_tests(size['NUM_OF_REGULAR_FULL_TESTS'], generate_regular_case, True, False)

        print(f"\n\nPerforming {size['NUM_OF_LARGE_TESTS']} large valid input tests on C code (Step 4 of 5):\n")
        perform_tests(size['NUM_OF_LARGE_TESTS'], generate_large_case, False, False)

        print(f"\n\nPerforming {size['NUM_OF_LARGE_FULL_TESTS']} large full valid input tests on C code (Step 5 of 5):\n")
        perform_tests(size['NUM_OF_LARGE_FULL_TESTS'], generate_large_case, True, False)

    except TestFailed as e:
        write_vectors(e.case.vectors, VECTORS_FILE_PATH)
        print(f"\nParameters Used: k={e.case.k}, iter={e.case.num_iter}.")
        print(f"Input used is saved to {VECTORS_FILE_PATH}")
    else:
        print ("\nTest Complete!")
        if size == SHORT_TEST_SIZE:
            print("\nFor a more extensive test run 'python kmeanstester.py long'")


if __name__ == '__main__':
    main()
