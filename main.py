"""
Perform Feature Engineering
"""
# Imports
import os
import json
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from utils import is_categorical
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit
import subprocess
from llmfe.logging import SplitLogger

# Arguments
parser = ArgumentParser()
parser.add_argument('--port', type=int, default=None)
parser.add_argument('--use_api', type=bool, default=False)
parser.add_argument('--api_model', type=str, default="gpt-3.5-turbo")
parser.add_argument('--spec_path', type=str)
parser.add_argument('--log_path', type=str, default="./logs/oscillator1")
parser.add_argument('--problem_name', type=str, default="oscillator1")
parser.add_argument('--run_id', type=int, default=1)
args = parser.parse_args()


if __name__ == '__main__':
    # Define the maximum number of iterations
    global_max_sample_num = 200
    splits = 5
    seed = 42
    # Load prompt specification
    with open(
        os.path.join(args.spec_path),
        encoding="utf-8",
    ) as f:
        specification = f.read()

    problem_name = args.problem_name
    label_encoder = preprocessing.LabelEncoder()
    is_regression = False
    if problem_name in ['btc','forest-fires', 'housing', 'insurance', 'bike', 'wine', 'crab']:
        is_regression = True
        print("Regression Accepted")

    # Load data observations
    file_name = f"./data/{problem_name}.csv"
    df = pd.read_csv(file_name)
    
    target_attr = df.columns[-1]
    # Alteration for data cat
    meta_data_name = f"./data/{problem_name}-metadata.json"
    meta_data = {}
    try:
        with open(meta_data_name, "r") as f:
            meta_data = json.load(f)
    except Exception as e:
        print(f"Warning: failed to load metadata from {meta_data_name}: {e}")
    # End alteration for data cat 

    attribute_names = df.columns[:-1].tolist()
    is_cat_map = {
        name: is_categorical(df[name], meta_data, name)
        for name in attribute_names
    }
    X = df.convert_dtypes()
    y = df[target_attr].to_numpy()
    label_list = np.unique(y).tolist()

    X = X.drop(target_attr, axis=1)
    date_column_name = 'date' # ensure date column is not included 
    if date_column_name in X.columns:
        X = X.drop(columns=[date_column_name])

    for col in X.columns:
        if X[col].dtype == 'string':
            X[col] = label_encoder.fit_transform(X[col])


    # Handle missing values
    X = X.fillna(0)
    if is_regression == False:
        y = label_encoder.fit_transform(y)
    else:
        y = y
 
    # Load metadata
    meta_data_name = f"./data/{problem_name}-metadata.json"
    meta_data = {}
    try:
        with open(meta_data_name, "r") as f:
            filed_meta_data = json.load(f)
        if isinstance(filed_meta_data, dict) and (
                'continuous_features' in filed_meta_data or
                'categorical_features' in filed_meta_data):
            for feature in filed_meta_data.get('continuous_features', []):
                name = feature.get('name')
                if not name:
                    continue
                meta_data[name] = {
                    "type": feature.get('type', 'continuous'),
                    "description": feature.get('description', ''),
                    "context": feature.get('context', '')
                }
            for feature in filed_meta_data.get('categorical_features', []):
                name = feature.get('name')
                if not name:
                    continue
                meta_data[name] = {
                    "type": feature.get('type', 'categorical'),
                    "description": feature.get('description', ''),
                    "context": feature.get('context', '')
                }
        else:
            meta_data = filed_meta_data if isinstance(filed_meta_data, dict) else {}
    except Exception as e:
        print(f"Warning: failed to load metadata from {meta_data_name}: {e}")
    
    tscv = TimeSeriesSplit(n_splits=splits) if is_regression else StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    print(tscv)
    review_processes = []
    i = 0
    for train_idx, test_idx in tscv.split(X, y):
        # Load config and parameters
        from llmfe import config
        from llmfe import sampler
        from llmfe import evaluator
        from llmfe import pipeline

        class_config = config.ClassConfig(llm_class=sampler.LocalLLM, sandbox_class=evaluator.LocalSandbox)
        config = config.Config(use_api = args.use_api,
                            api_model = args.api_model,)
        X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        i +=1

        current_is_cat = [is_cat_map.get(col, False) for col in X.columns]
        data_dict = {'inputs': X_train_fold, 'outputs': y_train_fold, 'is_cat': current_is_cat, 'is_regression': is_regression}
        dataset = {'data': data_dict}
        log_path = args.log_path + f"_split_{i}"
        
        # New prompt logging and evaluation functionality
        # Start logger
        split_logger = SplitLogger(run_id=str(args.run_id), split_idx=str(i))
        current_log_file = split_logger.log_file_path
        print(f'--- Split {i}: Logging reviews to {current_log_file} ---')
        
        # Run pipeline and capture iID

        wandb_run_id = pipeline.main(
            specification=specification,
            inputs=dataset,
            config=config,
            meta_data=meta_data,
            max_sample_nums=global_max_sample_num*splits,
            class_config=class_config,
            log_dir=log_path,
            logger=split_logger
        )
        
        # Close logger and send to gemini
        split_logger.close()
        
        api_key = os.environ.get('GEMINI_API_EVALUATOR') 
        review_model = 'gemini-2.5-flash'
        
        if api_key and wandb_run_id:
            print(f'Triggering Gemini Review for Run ID: {wandb_run_id}')
            p=subprocess.Popen([
                'python',
                'analysis/program_review.py',
                '--input_file', current_log_file,
                '--api_key', api_key,
                '--wandb_run_id', wandb_run_id,
                '--model', review_model,
                "--metadata_path",meta_data_name
            ])
            review_processes.append(p)
        else:
            
            print("Skipping review dueto missing API Key or WandB Run ID.")
        
    print("Training loop finished. Waiting for all review uploads to complete...")
    for p in review_processes:
        p.wait()
    print("All processes finished. Exiting.")    
