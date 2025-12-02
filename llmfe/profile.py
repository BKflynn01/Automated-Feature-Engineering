# profile the experiment with tensorboard

from __future__ import annotations

import os.path
from typing import List, Dict
import logging
import json
import html
import wandb
from wandb.errors import AuthenticationError, CommError
from requests.exceptions import RequestException
from llmfe import code_manipulation
from torch.utils.tensorboard import SummaryWriter
from llmfe.buffer import _get_signature, _reduce_score
from wandb import Table, plot
from collections import defaultdict, Counter


class Profiler:
    def __init__(
            self,
            log_dir: str | None = None,
            pkl_dir: str | None = None,
            max_log_nums: int | None = None,
            wandb_enable: bool = True,
            wandb_project: str = "llmfe-feature-engineering-btc-original",
            wandb_run_name: str | None = None,
            wandb_group_name: str | None = None,
            split_id: int | None = None,
            run_config: dict | None = None,
            base_step: int = 0
    ):
        """
        Args:
            log_dir     : folder path for tensorboard log files.
            pkl_dir     : save the results to a pkl file.
            max_log_nums: stop logging if exceeding max_log_nums.
        """
        logging.getLogger().setLevel(logging.INFO)
        self._log_dir = log_dir
        self._json_dir = os.path.join(log_dir, 'samples')
        os.makedirs(self._json_dir, exist_ok=True)
        self._max_log_nums = max_log_nums
        self._num_samples = 0
        self._cur_best_program_sample_order = None
        self._cur_best_program_score = -99999999
        self._cur_best_program_str = None
        self._evaluate_success_program_num = 0
        self._evaluate_failed_program_num = 0
        self._tot_sample_time = 0
        self._tot_evaluate_time = 0
        self._all_sampled_functions: Dict[int, code_manipulation.Function] = {}
        
        self._split_id = split_id
        self._base_step = base_step
        self._use_wandb = bool(wandb_enable)

        if log_dir:
            self._writer = SummaryWriter(log_dir=log_dir)

        self._each_sample_best_program_score = []
        self._each_sample_evaluate_success_program_num = []
        self._each_sample_evaluate_failed_program_num = []
        self._each_sample_tot_sample_time = []
        self._each_sample_tot_evaluate_time = []
        self._cluster_counts: Dict[int, Counter] = defaultdict(Counter)
            
        if self._use_wandb:
            if wandb_run_name is not None:
                _run_name = wandb_run_name
            elif log_dir is not None:
                _run_name = log_dir
            else:
                _run_name ="run"
                
            wandb.init(project=wandb_project,
                       name=_run_name,
                       group=wandb_group_name,
                       config=run_config,
                       reinit=True)
            wandb.define_metric("global_step")
            wandb.define_metric('*', step_metric="global_step")
            self._wb_table = Table(columns=[
                "global_step",
                "split_id",
                "island_id",
                "score",
                "sample_time",
                "evaluate_time",
                "cluster_signature",
                "cluster_reduced_score",
                "function_str",
                "prompt_id",
                "head_type",
                "version_generated",
            ])
            self._wb_prompt_table = Table(columns=[
                "prompt_step",
                "split_id",
                "island_id",
                "prompt_id",
                "version_generated",
                "head_type",
                "num_samples"
            ])
            
                
                       
    def _write_tensorboard(self):
        if not self._log_dir:
            return

        self._writer.add_scalar(
            'Best Score of Function',
            self._cur_best_program_score,
            global_step=self._num_samples
        )
        self._writer.add_scalars(
            'Legal/Illegal Function',
            {
                'legal function num': self._evaluate_success_program_num,
                'illegal function num': self._evaluate_failed_program_num
            },
            global_step=self._num_samples
        )
        self._writer.add_scalars(
            'Total Sample/Evaluate Time',
            {'sample time': self._tot_sample_time, 'evaluate time': self._tot_evaluate_time},
            global_step=self._num_samples
        )
        
        # Log the function_str
        self._writer.add_text(
            'Best Function String',
            self._cur_best_program_str,
            global_step=self._num_samples
        )
    def log_prompt(self, 
                   *,
                   prompt_id: str, 
                   island_id: int, 
                   version_generated: int, 
                   prompt_code: str, 
                   num_samples: int, 
                   step_hint: int | None = None,
                   head_type: str | None = None,
                   instruction_prompt: str | None = None):
        # Log to wandb prompt table
        if step_hint is None: 
            _step_int = 0
        else:
            _step_hint = int(step_hint)
        prompt_step = self._base_step + _step_hint
        if self._use_wandb: self._wb_prompt_table.add_data(
            prompt_step, 
            self._split_id, 
            island_id,
            prompt_id,  
            int(version_generated),
            head_type,
            int(num_samples),
        )
        if self._use_wandb:
            combined_prompt = prompt_code
            if instruction_prompt:
                combined_prompt = "\n".join([instruction_prompt, prompt_code])
            _prev = combined_prompt.replace("<", "&lt;").replace(">","&gt;")
            wandb.log({"Prompts/text": wandb.Html(f"<pre>{_prev}</pre>")}, step=prompt_step)
    def _log_wandb(self, 
                   *,
                   programs: code_manipulation.Function,
                   island_id: int | None, 
                   scores_per_test: dict | None,
                   prompt_code: str | None = None):
        if not self._use_wandb:
            return 
        if getattr(programs, "global_sample_nums", None) is None:
                   step_local = 0
        else: 
            step_local = int(programs.global_sample_nums)
        global_step = self._base_step + step_local
        function_str = str(programs).strip("\n")
        score = programs.score
        sample_time = programs.sample_time
        evaluate_time = programs.evaluate_time
        prompt_id = getattr(programs,"prompt_id", None)
        version_generated = getattr(programs, "version_generated", None)
        head_type = getattr(programs, "head_type", None)
        cluster_signature = None
        cluster_reduced_score = None
        if scores_per_test:
            sig_tuple = _get_signature(scores_per_test)
            cluster_signature = str(tuple(f"{s:.4f}" for s in sig_tuple))
            signature_value = float(f"{sig_tuple[0]:.6f}") if sig_tuple else None
            try:
                cluster_reduced_score = float(_reduce_score(scores_per_test))
            except Exception:
                pass
        else:
            signature_value = None
        
        log_data = {
            "global_step": global_step,
            "split_id": self._split_id,
            "Run_Metrics/score": score,
            "Run_Metrics/Sample_time": sample_time,
            "Run_Metrics/Evaluate_Time": evaluate_time,
            "Run_Metrics/Number_Samples": self._num_samples,
            "Clusters/Cluster_Signatures": cluster_reduced_score,
            **({f"Island/{island_id}_score": score} if island_id is not None and score is not None else {}),
        }
        
        try:
            wandb.log(log_data, step=global_step)
        except (AuthenticationError, CommError, RequestException, TimeoutError) as exc:
            logging.warning("W&B metric log failed, skipping this step: %s", exc)
            return
        if self._use_wandb and island_id is not None:
            self._log_program_media(
                island_id=island_id,
                step=global_step,
                program_str=function_str,
                score=score,
                prompt_id=prompt_id,
            )
        self._wb_table.add_data(
            global_step, 
            self._split_id,
            island_id,
            score,
            sample_time,
            evaluate_time,
            cluster_signature,
            cluster_reduced_score,
            function_str,
            prompt_id,
            head_type,
            version_generated,
        )    
        
        if self._use_wandb and island_id is not None and signature_value is not None:
            try:
                self._log_cluster_histogram(island_id=island_id,
                                            signature_value=signature_value,
                                            step=global_step)
            except (AuthenticationError, CommError, RequestException, TimeoutError) as exc:
                logging.warning("W&B histogram log failed: %s", exc)

    def _log_cluster_histogram(self, *, island_id: int, signature_value: float, step: int) -> None:
        counter = self._cluster_counts[island_id]
        counter[signature_value] += 1
        table = Table(columns=["signature", "count"])
        for sig, count in sorted(counter.items()):
            table.add_data(float(sig), int(count))
        chart = plot.bar(
            table,
            "signature",
            "count",
            title=f"Island {island_id} Cluster Histogram"
        )
        try:
            wandb.log({f"Clusters/{island_id}/Histogram": chart}, step=step)
        except (AuthenticationError, CommError, RequestException, TimeoutError) as exc:
            logging.warning("Failed to log cluster histogram to W&B: %s", exc)

    def _log_program_media(self, *, island_id: int, step: int, program_str: str, score: float | None, prompt_id: str | None) -> None:
        escaped_program = html.escape(program_str)
        header_items = [
            f"Global Step: {step}",
            f"Island ID: {island_id}",
        ]
        if score is not None:
            header_items.append(f"Score: {score}")
        if prompt_id:
            header_items.append(f"Prompt ID: {prompt_id}")
        header_html = "<br/>".join(header_items)
        body = f"<div><strong>{header_html}</strong><pre>{escaped_program}</pre></div>"
        try:
            wandb.log({f"Programs/Island_{island_id}": wandb.Html(body)}, step=step)
        except (AuthenticationError, CommError, RequestException, TimeoutError) as exc:
            logging.warning("Failed to log program HTML to W&B: %s", exc)
            
    def _write_json(self, programs: code_manipulation.Function, island_id: int | None, scores_per_test: dict | None):
        sample_order = programs.global_sample_nums
        sample_order = sample_order if sample_order is not None else 0
        function_str = str(programs)
        score = programs.score
        
        cluster_signature = None
        sig = _get_signature(scores_per_test) if scores_per_test else ()
        cluster_signature = sig[0] if sig else None 
            

        content = {
            'sample_order': sample_order,
            'island_id': island_id,
            'cluster_signature': cluster_signature,
            'function': function_str,
            'score': score
        }
        path = os.path.join(self._json_dir, f'samples_{sample_order}.json')
        with open(path, 'w') as json_file:
            json.dump(content, json_file, indent=2)

    def register_function(self, programs: code_manipulation.Function, island_id: int | None = None, scores_per_test: dict | None = None, **kwargs):
        if self._max_log_nums is not None and self._num_samples >= self._max_log_nums:
            return

        sample_orders: int = programs.global_sample_nums
        if sample_orders not in self._all_sampled_functions:
            self._num_samples += 1
            self._all_sampled_functions[sample_orders] = programs
            self._record_and_verbose(sample_orders)
            self._write_tensorboard()
            self._write_json(programs, island_id, scores_per_test)
            self._log_wandb(programs=programs, island_id=island_id, scores_per_test=scores_per_test, prompt_code=kwargs.get('prompt_code'))

    def _record_and_verbose(self, sample_orders: int):
        function = self._all_sampled_functions[sample_orders]
        function_str = str(function).strip('\n')
        sample_time = function.sample_time
        evaluate_time = function.evaluate_time
        score = function.score
        # log attributes of the function
        print(f'================= Evaluated Function =================')
        print(f'{function_str}')
        print(f'------------------------------------------------------')
        print(f'Score        : {str(score)}')
        print(f'Sample time  : {str(sample_time)}')
        print(f'Evaluate time: {str(evaluate_time)}')
        print(f'Sample orders: {str(sample_orders)}')
        print(f'======================================================\n\n')

        # update best function in curve
        if function.score is not None and score > self._cur_best_program_score:
            self._cur_best_program_score = score
            self._cur_best_program_sample_order = sample_orders
            self._cur_best_program_str = function_str

        # update statistics about function
        if score:
            self._evaluate_success_program_num += 1
        else:
            self._evaluate_failed_program_num += 1

        if sample_time:
            self._tot_sample_time += sample_time
        if evaluate_time:
            self._tot_evaluate_time += evaluate_time
            
            
    def close(self):
        if hasattr(self,'_writer'):
                   self._writer.flush()
                   self._writer.close()
        if self._use_wandb:
            wandb.run.summary["num_samples"] = self._num_samples
            wandb.run.summary["split_id"] = self._split_id
            wandb.log({"proposals/table": self._wb_table})
            wandb.log({"prompts/table": self._wb_prompt_table})
            wandb.finish()
