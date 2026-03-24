import json
import os
import time
from typing import Any


class SplitLogger:
    def __init__(self, run_id: str, split_idx: int, log_dir: str = "logs/reviews"):
        """
        Init logger
        """
        # Ensure directory exists
        os.makedirs(log_dir, exist_ok=True)
        self.timestamp = int(time.time())

        # Filename format: <run_id>_split_<split_idx>_<TIMESTAMP>.jsonl
        filename = f"{run_id}_split_{split_idx}_{self.timestamp}.jsonl"
        self.log_file_path = os.path.join(log_dir, filename)

        # Open in append mode
        self.file_handle = open(self.log_file_path, "a")

    def extract_code_only(self, program_text: str) -> str:
        """
        Helper function to extract feature altering code from the proprosed program
        """
        code = []
        for line in program_text.split("\n"):
            stripped_line = line.strip()
            if stripped_line.startswith("df_output["):
                code.append(line)
        return "\n".join(code)

    def log_program(self, program: str, score: Any, prompt_id: str = None):
        """
        Append program to log file
        """
        if hasattr(score, "item"):
            score = score.item()
        cleaned_program = self.extract_code_only(program)
        log_entry = {
            "timestamp": int(time.time()),
            "prompt_id": prompt_id,
            "generated_program": cleaned_program,
            "score": score,
        }
        json.dump(log_entry, self.file_handle)
        self.file_handle.write("\n")
        self.file_handle.flush()

    def close(self):
        """
        Close the file
        """
        if self.file_handle:
            self.file_handle.close()

    def get_log_path(self) -> str:
        """
        Return the full path to the current log file
        """
        return self.log_file_path
