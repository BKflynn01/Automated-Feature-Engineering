# LLM-FE: Automated Feature Engineering for Tabular Data with LLMs as Evolutionary Optimizers
[![arXiv](https://img.shields.io/badge/arXiv-2503.14434-b31b1b.svg)](https://arxiv.org/abs/2503.14434)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-LLMFE-yellow)](https://huggingface.co/papers/2503.14434)

Official implementation of  [LLM-FE: Automated Feature Engineering for Tabular Data with LLMs as Evolutionary Optimizers.](https://arxiv.org/abs/2503.14434)

![](llmfe.jpg)

## 📄 Overview
LLM-FE is a novel framework that leverages Large Language Models (LLMs) as evolutionary optimizers to automate feature engineering for tabular datasets.  LLM-FE iteratively generates and refines features using structured prompts, selecting high-impact transformations based on model performance. This approach enables the discovery of interpretable and high-quality features, enhancing the performance of various machine learning models across diverse classification and regression tasks.

## Current Improvements 
- [x] Integrate WandB for deeper evaluation.
- [x] Add support for Gemini and GPT-5
- [x] Integrate API (Gemini) for feature evaluation.
- [x] Modify prompt with score feedback.
- [x] Modify metadata and utlis.py for correct data classification.
- [x] Modify to handle time series data.
- [ ] Utilize Genetic Algorithm for feature credit assignment to generate the optimal set of features.

## Before Running 
This project requires access to LLM providers and experiment tracking. You will need to generate API keys for the following services:

* **Google Gemini:**
    * Go to [Google AI Studio](https://aistudio.google.com/app/apikey).
* **OpenAI:**
    * Go to [OpenAI Platform](https://platform.openai.com/api-keys).
* **Weights & Biases (WandB):**
    * Sign up for a free account at [wandb.ai](https://wandb.ai/site).
    * Go to your [Settings/API Keys](https://wandb.ai/authorize) to copy your personal API key.
    * After running provide your API key in the terminal.
    
## ⚙️ Installation
To run the code, create a conda environment and install the dependencies using `requirements.txt`:

```
conda create -n llmfe python=3.11.7
conda activate llmfe
pip install -r requirements.txt
```

## ✅ Auto Lint/Format Before Push
To automatically run formatting and lint checks before code leaves your machine:

```
pip install -r requirements-dev.txt
pre-commit install --hook-type pre-push
pre-commit run --all-files --hook-stage pre-push
```

What this repository hook setup does:
- Runs `ruff check tests ga_optimizer utils.py` before each push.
- Runs `ruff format --check tests ga_optimizer utils.py` before each push.
- Runs `mypy --config-file pyproject.toml` before each push.
- Runs `bandit -c pyproject.toml -r ga_optimizer` and `bandit -c pyproject.toml utils.py` before each push (`B102`/`exec` check skipped).
- Runs `pytest` (with coverage) before each push.

## 🔧 Usage
Set the API key in run_llmfe.sh:
```
export API_KEY= <OpenAI API Key>
export GEMINI_API_KEY=<Gemini API Key>
export GEMINI_API_EVALUATOR=<Gemini API Key>
```
Uncomment the desired prediction problem in run_llmfe.sh and set the desired API Model:
```
python main.py --use_api True --api_model "gpt-3.5-turbo" --problem_name btc --spec_path ./specs/specification_btc.txt --log_path ./logs/btc_gpt_3.5_turbo
```
To run the LLM-FE pipeline on a sample dataset:
```
bash run_llmfe.sh
```

## 📝 Citation
```
@article{abhyankar2025llm,
  title={LLM-FE: Automated Feature Engineering for Tabular Data with LLMs as Evolutionary Optimizers},
  author={Abhyankar, Nikhil and Shojaee, Parshin and Reddy, Chandan K},
  journal={arXiv preprint arXiv:2503.14434},
  year={2025}
}
```

## 📄 License

This repository is licensed under MIT licence.

This work is built on top of other open source projects like [FunSearch](https://github.com/google-deepmind/funsearch) and [LLM-SR](https://github.com/deep-symbolic-mathematics/llm-sr). We thank the original contributors of these works for open-sourcing their valuable source codes.


## 📬 Contact Us
For any questions or issues, you are welcome to open an issue in this repo, or contact us at  [nikhilsa@vt.edu](nikhilsa@vt.edu) and [parshinshojaee@vt.edu](parshinshojaee@vt.edu).
