"""Class for evaluating programs proposed by the Sampler."""

from __future__ import annotations

import ast
import copy
import profile
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Type

import pandas as pd

from llmfe import buffer, code_manipulation, evaluator_accelerate


class _FunctionLineVisitor(ast.NodeVisitor):
    """Visitor that finds the last line number of a function with a given name."""

    def __init__(self, target_function_name: str) -> None:
        self._target_function_name: str = target_function_name
        self._function_end_line: int | None = None

    def visit_FunctionDef(self, node: Any) -> None:
        """Collect the end line number of the target function."""
        if node.name == self._target_function_name:
            self._function_end_line = node.end_lineno
        self.generic_visit(node)

    @property
    def function_end_line(self) -> int:
        """Line number of the final line of function `target_function_name`."""
        assert self._function_end_line is not None
        return self._function_end_line


def _trim_function_body(generated_code: str) -> str:
    """Extract the body of the generated function, trimming anything after it.
    Please note that the indentation is REQUIRED !!!
    """
    if not generated_code:
        return ""

    code = f"def fake_function_header():\n{generated_code}"

    tree = None
    while tree is None:
        try:
            tree = ast.parse(code)

        except SyntaxError as e:
            if e.lineno is None:  # Nothing could be saved when syntaxError
                return ""
            code = "\n".join(code.splitlines()[: e.lineno - 1])

    if not code:
        return ""

    visitor = _FunctionLineVisitor("fake_function_header")
    visitor.visit(tree)
    body_lines = code.splitlines()[1 : visitor.function_end_line]
    return "\n".join(body_lines) + "\n\n"


def _sample_to_program(
    generated_code: str,
    version_generated: int | None,
    template: code_manipulation.Program,
    function_to_evolve: str,
) -> tuple[code_manipulation.Function, str]:
    """
    Return the compiled generated function and the full runnable program.
    This function removes the content after the generated function body.
    """
    body = _trim_function_body(generated_code)
    if version_generated is not None:
        body = code_manipulation.rename_function_calls(
            code=body,
            source_name=f"{function_to_evolve}_v{version_generated}",
            target_name=function_to_evolve,
        )

    program = copy.deepcopy(template)
    evolved_function = program.get_function(function_to_evolve)
    evolved_function.body = body

    return evolved_function, str(program)


class Sandbox(ABC):
    """Sandbox for executing generated code."""

    @abstractmethod
    def run(
        self,
        program: str,
        function_to_run: str,
        function_to_evolve: str,
        inputs: Any,
        test_input: str,
        timeout_seconds: int,
        **kwargs,
    ) -> tuple[Any, bool]:
        """Return `function_to_run(test_input)` and whether execution succeeded."""
        raise NotImplementedError(
            "Must provide a sandbox for executing untrusted code."
        )


class LocalSandbox(Sandbox):
    """
    Secure environment for executing and evaluating LLM generated programs.
    Prevents harmful operations, limits resource usage, and enforces timeouts.
    Returns a 'score' for the executed program.
    """

    def __init__(self, verbose=False, numba_accelerate=False):
        """
        Initialize Sandbox.

        Args:
        verbose (bool): Enable detailed output.
        numba_accelerate (bool): Use Numba for acceleration of evaluation (limited compatibility).
        """
        self._verbose = verbose
        self._numba_accelerate = numba_accelerate

    def run(
        self,
        program: str,
        function_to_run: str,
        function_to_evolve: str,
        inputs: Any,
        test_input: str,
        timeout_seconds: int,
        **kwargs,
    ) -> tuple[Any, bool]:
        """
        Execute the given program sample and return its score and success status.

        Note: This sandbox is specific to the equation program skeleton discovery problem.
        """

        dataset = inputs[test_input]
        result_container = []
        results = self._compile_and_run_function(
            program,
            function_to_run,
            function_to_evolve,
            dataset,
            self._numba_accelerate,
            result_container,
        )

        # # Wait for the computation to finish or timeout
        # computation_thread.join(timeout=timeout_seconds)

        if self._verbose:
            self._print_evaluation_details(program, results, **kwargs)
        return results

    def _get_results(self, queue):
        for _ in range(5):
            if not queue.empty():
                return queue.get_nowait()
            time.sleep(0.1)
        return None, False

    def _print_evaluation_details(self, program, results, **kwargs):
        print("================= Evaluated Program =================")
        function = code_manipulation.text_to_program(program).get_function(
            kwargs.get("func_to_evolve", "equation")
        )
        print(
            f"{str(function).strip()}\n-----------------------------------------------------"
        )
        print(
            f"Score: {results}\n=====================================================\n\n"
        )

    def _compile_and_run_function(
        self,
        program,
        function_to_run,
        function_to_evolve,
        dataset,
        numba_accelerate,
        result_container,
    ):
        try:
            # optimize the code (decorate function_to_run with @numba.jit())
            if numba_accelerate:
                program = evaluator_accelerate.add_numba_decorator(
                    program=program, function_to_evolve=function_to_evolve
                )

            # execute the program, map func/var/class to global namespace
            all_globals_namespace = {}
            exec(program, all_globals_namespace)
            function_to_run = all_globals_namespace[function_to_run]
            results = function_to_run(dataset)

            if not isinstance(results[0], (int, float)):
                result_container.append(None)
                result_container.append(False)
            else:
                result_container.append(results)
                result_container.append(True)
        # if raise any exception, execution is failed
        except Exception as e:
            print(f"Execution Error: {e}")
            result_container.append(None)
            result_container.append(False)

        return tuple(result_container)


def _calls_ancestor(program: str, function_to_evolve: str) -> bool:
    """Return whether the generated function is calling an earlier version."""
    for name in code_manipulation.get_functions_called(program):
        if name.startswith(f"{function_to_evolve}_v"):
            return True
    return False


_TIME_SERIES_LEAKAGE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\.shift\(\s*-\d+", "shift with negative periods"),
    (r"\.shift\(\s*periods\s*=\s*-\d+", "shift(periods=<negative>)"),
    (r"\.diff\(\s*-\d+", "diff with negative periods"),
    (r"\.diff\(\s*periods\s*=\s*-\d+", "diff(periods=<negative>)"),
    (r"\.pct_change\(\s*-\d+", "pct_change with negative periods"),
    (r"\.pct_change\(\s*periods\s*=\s*-\d+", "pct_change(periods=<negative>)"),
    (r"\.rolling\([^)]*center\s*=\s*True", "centered rolling window"),
    (r"\.fillna\([^)]*method\s*=\s*['\"]bfill['\"]", "fillna(method='bfill')"),
    (r"\.bfill\(", "backward fill"),
)


def _detect_time_series_leakage(function_body: str) -> list[str]:
    normalized = function_body.replace("\n", " ")
    findings: list[str] = []
    for pattern, reason in _TIME_SERIES_LEAKAGE_PATTERNS:
        if re.search(pattern, normalized):
            findings.append(reason)
    return findings


class Evaluator:
    """Class that analyses functions generated by LLMs."""

    def __init__(
        self,
        database: buffer.ExperienceBuffer,
        template: code_manipulation.Program,
        function_to_evolve: str,
        function_to_run: str,
        inputs: Sequence[Any],
        timeout_seconds: int = 30,
        sandbox_class: Type[Sandbox] = Sandbox,
        logger: Any = None,
    ):
        self._database = database
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._function_to_run = function_to_run
        self._inputs = inputs
        self._timeout_seconds = timeout_seconds
        self._sandbox = sandbox_class()
        self._logger = logger

    def analyse(
        self,
        sample: str,
        island_id: int | None,
        data_input: pd.DataFrame | None,
        data_output: pd.DataFrame | None,
        version_generated: int | None,
        **kwargs,
    ) -> None:
        """Compile the hypothesis sample into a program and executes it on test inputs."""
        new_function, program = _sample_to_program(
            sample, version_generated, self._template, self._function_to_evolve
        )
        is_time_series = bool(self._inputs.get("data", {}).get("is_time_series", False))
        leakage_findings = (
            _detect_time_series_leakage(new_function.body) if is_time_series else []
        )
        if leakage_findings:
            print(
                f"Rejected sample due to potential time-series leakage: {', '.join(leakage_findings)}"
            )
            return
        scores_per_test = {}
        eval_metrics_per_test = {}
        time_reset = time.perf_counter()
        self._inputs["data"]["inputs"] = data_input
        self._inputs["data"]["outputs"] = data_output
        for current_input in self._inputs:
            test_output, runs_ok = self._sandbox.run(
                program,
                self._function_to_run,
                self._function_to_evolve,
                self._inputs,
                current_input,
                self._timeout_seconds,
            )
            if (
                runs_ok
                and not _calls_ancestor(program, self._function_to_evolve)
                and test_output is not None
            ):
                if not isinstance(test_output[0], (int, float)):
                    print(f"Error: test_output is {test_output}")
                    raise ValueError("@function.run did not return an int/float score.")
                scores_per_test[current_input] = test_output[0]
                input_data = test_output[1]
                output_data = test_output[2]
                if len(test_output) >= 4 and isinstance(test_output[3], dict):
                    eval_metrics_per_test[current_input] = test_output[3]

        evaluate_time = time.perf_counter() - time_reset

        # To facilitate logging
        if self._logger:
            # Determine score to log. Possible it is none if program failed
            logged_score = None
            if scores_per_test:
                logged_score = list(scores_per_test.values())[0]
            self._logger.log_program(
                program=sample, score=logged_score, prompt_id=kwargs.get("prompt_id")
            )
        if scores_per_test:
            self._database.register_program(
                new_function,
                island_id,
                scores_per_test,
                **kwargs,
                input_data=input_data,
                output_data=output_data,
                eval_metrics_per_test=(
                    eval_metrics_per_test if eval_metrics_per_test else None
                ),
                evaluate_time=evaluate_time,
            )

        else:
            profiler: profile.Profiler = kwargs.get("profiler", None)
            if profiler:
                global_sample_nums = kwargs.get("global_sample_nums", None)
                sample_time = kwargs.get("sample_time", None)
                new_function.global_sample_nums = global_sample_nums
                new_function.score = None
                new_function.sample_time = sample_time
                new_function.evaluate_time = evaluate_time
                profiler.register_function(new_function)
