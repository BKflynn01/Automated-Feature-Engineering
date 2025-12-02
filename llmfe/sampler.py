""" Class for sampling new programs. """
from __future__ import annotations
from abc import ABC, abstractmethod

from typing import Collection, Sequence, Type
import numpy as np
import time

from llmfe import evaluator
from llmfe import buffer
from llmfe import config as config_lib
import requests
import json
import http.client
import os
import uuid
import datetime


class LLM(ABC):
    def __init__(self, samples_per_prompt: int) -> None:
        self._samples_per_prompt = samples_per_prompt

    def _draw_sample(self, prompt: str) -> str:
        """ Return a predicted continuation of `prompt`."""
        raise NotImplementedError('Must provide a language model.')

    @abstractmethod
    def draw_samples(self, prompt: str) -> Collection[str]:
        """ Return multiple predicted continuations of `prompt`. """
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]



class Sampler:
    """ Node that samples program skeleton continuations and sends them for analysis. """
    _global_samples_nums: int = 0

    def __init__(
            self,
            database: buffer.ExperienceBuffer,
            evaluators: Sequence[evaluator.Evaluator],
            samples_per_prompt: int,
            meta_data: dict,
            config: config_lib.Config,
            max_sample_nums: int | None = None,
            llm_class: Type[LLM] = LLM,
    ):
        self._samples_per_prompt = samples_per_prompt
        self._database = database
        self._evaluators = evaluators
        self._meta_data = meta_data
        self._llm = llm_class(samples_per_prompt)
        self._max_sample_nums = max_sample_nums
        self.config = config
        self.__class__._global_samples_nums = 0

    
    def sample(self, **kwargs):
        """ Continuously gets prompts, samples programs, sends them for analysis. """
        profiler = kwargs.get('profiler', None)
        while True:
            # stop the search process if hit global max sample nums
            if self._max_sample_nums//5 and self.__class__._global_samples_nums >= self._max_sample_nums//5:
                break
            
            prompt = self._database.get_prompt()
            
            prompt_id = str(uuid.uuid4())
            head_type = "operatons" if "<Operators>" in prompt.code else "domain"
            instruction_prompt = getattr(self._llm, "_instruction_prompt", "")
            if profiler:
                profiler.log_prompt(
                    prompt_id = prompt_id,
                    island_id = prompt.island_id,
                    version_generated = prompt.version_generated,
                    prompt_code = prompt.code,
                    num_samples = self._samples_per_prompt,
                    step_hint = self._get_global_sample_nums(),
                    head_type = "operations" if "<Operators>" in prompt.code else "domain",
                    instruction_prompt=instruction_prompt
                )
            # DELETE
            
            os.makedirs("logs", exist_ok=True)

            head_type = "operations" if "<Operators>" in prompt.code else "domain"

            # Append to CSV
            with open("logs/run_log.csv", "a", encoding="utf-8") as f:
                f.write(f"\n=====================NEW=====================\nDATE:{datetime.datetime.now().isoformat()}\nPROMPT TYPE:{head_type}\n")

            # Dump the full prompt to a text log 
            with open("logs/prompts.log.txt", "a", encoding="utf-8") as f:
                f.write("\n================ PROMPT ================\n")
                f.write(f"ts: {datetime.datetime.now().isoformat()}\n")
                #f.write(f"island_id: {prompt.island_id} | head_type: {head_type}  \n")
                f.write("----------------------------------------\n")
                f.write(prompt.code if isinstance(prompt.code, str) else str(prompt.code))
                f.write("\n========================================\n")
            
            
            
            # END OF DELETE
            
            reset_time = time.time()
            samples = self._llm.draw_samples(prompt.code,self.config)
            sample_time = (time.time() - reset_time) / self._samples_per_prompt

            # This loop can be executed in parallel on remote evaluator machines.
            for sample in samples:
                sample = "\n    import pandas as pd\n    import numpy as np\n" + sample
                self._global_sample_nums_plus_one()
                cur_global_sample_nums = self._get_global_sample_nums()
                chosen_evaluator: evaluator.Evaluator = np.random.choice(self._evaluators)
                chosen_evaluator.analyse(
                    sample,
                    island_id=prompt.island_id,
                    data_input=prompt.data_input,
                    data_output=prompt.data_output,
                    version_generated=prompt.version_generated,
                    global_sample_nums=cur_global_sample_nums,
                    sample_time=sample_time,
                    profiler=kwargs.get('profiler', None),
                    prompt_id=prompt_id, #altered
                    head_type = head_type,
                    prompt_code=prompt.code
                )

    def _get_global_sample_nums(self) -> int:
        return self.__class__._global_samples_nums

    def set_global_sample_nums(self, num):
        self.__class__._global_samples_nums = num

    def _global_sample_nums_plus_one(self):
        self.__class__._global_samples_nums += 1



def _extract_body(sample: str, config: config_lib.Config) -> str:
    """
    Extract the function body from a response sample, removing any preceding descriptions
    and the function signature. Preserves indentation.
    ------------------------------------------------------------------------------------------------------------------
    Input example:
    ```
    This is a description...
    def function_name(...):
        return ...
    Additional comments...
    ```
    ------------------------------------------------------------------------------------------------------------------
    Output example:
    ```
        return ...
    Additional comments...
    ```
    ------------------------------------------------------------------------------------------------------------------
    If no function definition is found, returns the original sample.
    """
    lines = sample.splitlines()
    func_body_lineno = 0
    find_def_declaration = False
    
    for lineno, line in enumerate(lines):
        # find the first 'def' program statement in the response
        if line[:3] == 'def':
            func_body_lineno = lineno
            find_def_declaration = True
            break
    
    if find_def_declaration:
        # for gpt APIs
        if config.use_api:
            code = ''
            for line in lines[func_body_lineno + 1:]:
                if "This program scored:" not in line:
                    code += line + '\n'
         
        # for mixtral
        else:
            code = ''
            indent = '    '
            for line in lines[func_body_lineno + 1:]:
                if "This program scored:" not in line:
                    if line[:4] != indent:
                        line = indent + line
                    code += line + '\n'
        
        return code
    
    return sample



class LocalLLM(LLM):
    def __init__(self, samples_per_prompt: int, batch_inference: bool = True, trim=True) -> None:
        """
        Args:
            batch_inference: Use batch inference when sample equation program skeletons. The batch size equals to the samples_per_prompt.
        """
        super().__init__(samples_per_prompt)

        url = "http://127.0.0.1:5000/completions"
        instruction_prompt_o = ("You are a helpful assistant tasked with discovering new features/ dropping less important feaures for the given prediction task. \
                             Complete the 'modify_features' function below, considering the physical meaning and relationships of inputs. Each program will be given a score as a reference.\n\n")

        instruction_prompt = ("You are a helpful assistant tasked with discovering new features/ dropping less important feaures for the given prediction task. \
                             Complete the 'modify_features' function below, considering the physical meaning and relationships of inputs.\n\n")
     
        self._batch_inference = batch_inference
        self._url = url
        self._instruction_prompt = instruction_prompt
        self._trim = trim


    def draw_samples(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        """Returns multiple equation program skeleton hypotheses for the given `prompt`."""
        if config.use_api:
            return self._draw_samples_api(prompt, config)
        else:
            return self._draw_samples_local(prompt, config)

    def _draw_samples_local(self, prompt: str, config: config_lib.Config) -> Collection[str]:    
        # instruction
        prompt = '\n'.join([self._instruction_prompt, prompt])
        while True:
            try:
                all_samples = []
                # response from llm server
                if self._batch_inference:
                    response = self._do_request(prompt)
                    for res in response:
                        all_samples.append(res)
                else:
                    for _ in range(self._samples_per_prompt):
                        response = self._do_request(prompt)
                        all_samples.append(response)

                # trim equation program skeleton body from samples
                if self._trim:
                    all_samples = [_extract_body(sample, config) for sample in all_samples]
                
                return all_samples
            except Exception:
                continue


    def _draw_samples_api(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        
        
        def _is_gpt5(name: str)-> bool:
            n = (name or "").lower()
            return n.startswith("gpt-5") 
        
        def _is_gemini(name: str) -> bool:
            n = (name or "").lower()
            return n.startswith("gemini")
        def _parse_responses_api(obj: dict) -> str:
            t = obj.get("output_text")
            if t:
                return t
            parts = []
            for item in obj.get("output", []):
                if item.get("type") == "message":
                    for c in item.get("content", []):
                        if c.get("type") == "output_text":
                            parts.append(c.get("text", ""))
            return "".join(parts)

        def _parse_gemini(obj: dict) -> str:
            ''' Program to parse responses from gemini'''
            try:
                cand = obj["candidates"][0]
                parts = cand.get("content", {}).get("parts", [])
                if parts:
                    return "".join(p.get("text", "") for p in parts)
                if "content" in cand and isinstance(cand["content"], dict):
                    return "".join(p.get("text", "") for p in cand["content"].get("parts", []))
            except Exception:
                pass
            return obj.get("candidates", [{}])[0].get("text", "") 
            
        all_samples = []
        prompt = '\n'.join([self._instruction_prompt, prompt])
        
        #DLETE AFTER
        os.makedirs("./logs/prompt_dumps", exist_ok=True)
        with open("./logs/prompt_dumps/full_prompt_11_18.txt", "a", encoding="utf-8") as f:
            f.write("\n====================== NEW PROMPT ======================\n")
            f.write(prompt if isinstance(prompt, str) else str(prompt))
            f.write("\n========================================================\n\n")
            
        model = getattr(config, "api_model", "")
        use_gpt5   = _is_gpt5(model)
        use_gemini = _is_gemini(model)

        # keys/env 
        openai_key  =  os.environ.get("API_KEY")
        google_key  =  os.environ.get("GEMINI_API_KEY")

        # Settings
        temperature = getattr(config, "temperature", 0.2)
        verbosity = getattr(config, "verbosity", "low") # "low" | "medium" | "high"
        reasoning_effort = getattr(config, "reasoning_effort", "low") # "minimal"|"medium"|"high"
        
        for _ in range(self._samples_per_prompt):
            attempts, backoff = 0,0.5
            while True:
                attempts +=1
                try:
                    if use_gemini:
                        if not google_key:
                            raise RuntimeError("Missing Gemini API key")
                        conn = http.client.HTTPSConnection("generativelanguage.googleapis.com")
                        payload = {
                        "contents": [
                            {"role": "user", "parts": [{"text": prompt}]}
                        ],
                        "generationConfig": {
                            "temperature": temperature,
                            "maxOutputTokens": 55000
                        }
                        }
                        path = f"/v1beta/models/{model}:generateContent?key={google_key}"
                        conn.request("POST", path, json.dumps(payload), {
                            "Content-Type": "application/json"
                        })
                        res = conn.getresponse()
                        raw = res.read().decode("utf-8")
                        if res.status >= 400:
                            raise RuntimeError(f"Gemini error {res.status}: {raw}")
                        
                        data = json.loads(raw)
                        #print("\n================= RAW GEMINI API RESPONSE ==================")
                        print(json.dumps(data, indent=2))
                        #print("=========================================================\n")
                        response_text = _parse_gemini(data)
                    
                    elif use_gpt5:
                    # -------- OpenAI GPT-5 (Responses API) --------
                        if not openai_key:
                            raise RuntimeError("Missing OPENAI_API_KEY")
                        conn = http.client.HTTPSConnection("api.openai.com")
                        payload = {
                                "model": model,
                                "input": prompt, 
                                "max_output_tokens": 10000, 
                                "text": {"verbosity": verbosity},
                                "reasoning": {"effort": reasoning_effort}
                            }
                        conn.request("POST", "/v1/responses", json.dumps(payload), {
                            "Authorization": f"Bearer {openai_key}",
                            "Content-Type": "application/json"
                        })
                        res = conn.getresponse()
                        raw = res.read().decode("utf-8")
                        if res.status >= 400:
                            raise RuntimeError(f"Responses API error {res.status}: {raw}")
                        data = json.loads(raw)
                        #print("\n================= RAW GPT-5 API RESPONSE ==================")
                        #print(data)
                        #print("=========================================================\n")
                        # ---------------------------------
                        data = json.loads(raw)
                        response_text = _parse_responses_api(data)

                    else:
                        # -------- OpenAI legacy Chat Completions --------
                        if not openai_key:
                            raise RuntimeError("Missing OPENAI_API_KEY")
                        conn = http.client.HTTPSConnection("api.openai.com")
                        payload = {
                            "model": model,  
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": 2000
                            
                        }
                        conn.request("POST", "/v1/chat/completions", json.dumps(payload), {
                            "Authorization": f"Bearer {openai_key}",
                            "Content-Type": "application/json"
                        })
                        res = conn.getresponse()
                        raw = res.read().decode("utf-8")
                        if res.status >= 400:
                            raise RuntimeError(f"Chat Completions error {res.status}: {raw}")
                        data = json.loads(raw)
                        #print("\n================= RAW GPT API RESPONSE ==================")
                        #print(data)
                        #print("=========================================================\n")
                        response_text = data["choices"][0]["message"]["content"]

                    if self._trim:
                        response_text = _extract_body(response_text, config)

                    all_samples.append(response_text or "    # empty response\n    pass\n")
                    break

                except Exception as e:
                    print(f"API call attempt {attempts} failed. Error: {e}") 
                    if attempts >= 6:
                        print("Max retries reached. Giving up on this sample.") 
                        all_samples.append("    # generation failed after retries\n    pass\n")
                        break
                    
                    backoff = min(backoff * 2, 8.0)
                    print(f"Retrying in {backoff:.1f} seconds...") 
                    time.sleep(backoff)

            return all_samples
                    
    '''   original
                    conn = http.client.HTTPSConnection("api.openai.com")
                    payload = json.dumps({
                        "max_tokens": 1500,
                        "model": config.api_model,
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    })
                    headers = {
                        'Authorization': f"Bearer {os.environ['API_KEY']}",
                        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                        'Content-Type': 'application/json'
                    }
                    conn.request("POST", "/v1/chat/completions", payload, headers)
                    res = conn.getresponse()
                    data = json.loads(res.read().decode("utf-8"))
                    response = data['choices'][0]['message']['content']
                    
                    if self._trim:
                        response = _extract_body(response, config)
                    
                    all_samples.append(response)
                    break

                except Exception:
                    continue
        
        return all_samples
    '''
    
    def _do_request(self, content: str) -> str:
        content = content.strip('\n').strip()
        # repeat the prompt for batch inference
        repeat_prompt: int = self._samples_per_prompt if self._batch_inference else 1
        
        data = {
            'prompt': content,
            'repeat_prompt': repeat_prompt,
            'params': {
                'do_sample': True,
                'temperature': None,
                'top_k': None,
                'top_p': None,
                'add_special_tokens': False,
                'skip_special_tokens': True,
            }
        }
        
        headers = {'Content-Type': 'application/json'}
        response = requests.post(self._url, data=json.dumps(data), headers=headers)
        
        if response.status_code == 200: #Server status code 200 indicates successful HTTP request! 
            response = response.json()["content"]
            
            return response if self._batch_inference else response[0]
