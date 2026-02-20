# ========================================
# 01.Data parsing & Preprocessing
# ========================================

import os
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy import stats
from scipy.special import softmax
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("Library import Complete")


# Data directory
DATA_DIR = "./routerbench_datasets"   # raw data file
WORK_DIR = "/content/routerbench_results"   # processed data file

# Preprocessing setting
MIN_MODELS = 3  # minimum response model number
MIN_QUERY_LENGTH = 10  # minimum query length
FILTER_NO_MODEL_CORRECT = True  # no_model_correct filtering
FILLNA_STRATEGY = 'keep_nan'  
RANDOM_STATE = 42  # random seed number

print("=" * 80)
print("RouterBench Data Preprocessing")
print("=" * 80)
print(f"\nPreprocessing setting:")
print(f"  minimum response model number:         {MIN_MODELS}")
print(f"  minimum query length character:            {MIN_QUERY_LENGTH}character")
print(f"  no_model_correct filtering:  {FILTER_NO_MODEL_CORRECT}")
print(f"  fillna strategy:              {FILLNA_STRATEGY}")
print(f"  random seed number:                 {RANDOM_STATE}")

# Working directory creation
os.makedirs(WORK_DIR, exist_ok=True)

# ========================================
# Phase 1: Raw-Data loading
# ========================================

print("\n" + "=" * 80)
print("Phase 1: Raw-Data loading")
print("=" * 80)

pkl_file = os.path.join(DATA_DIR, 'routerbench_0shot.pkl')
print(f"\nData loading: {pkl_file}")

with open(pkl_file, 'rb') as f:
    data = pickle.load(f)

print(f"Data load complete")
print(f"  Type: {type(data)}")

if isinstance(data, pd.DataFrame):
    print(f"  DataFrame shape: {data.shape}")
    print(f"  Column number: {len(data.columns)}")

# ========================================
# Phase 2: Separation of data by benchmark
# ========================================

print("\n" + "=" * 80)
print("Phase 2: Data structure parsing")
print("=" * 80)

benchmark_data = {}

if isinstance(data, pd.DataFrame):
    if 'eval_name' in data.columns:
        print(f"Format: Single DataFrame (separation by eval_name)")

        for eval_name in data['eval_name'].unique():
            # Exclude NaN or eval_name whicih is empty string
            if pd.isna(eval_name) or str(eval_name).strip() == '':
                continue
            benchmark_data[eval_name] = data[data['eval_name'] == eval_name].copy()
    else:
        print(f"Format: Single DataFrame (no separation)")
        benchmark_data['all_benchmarks'] = data

total_samples = sum(len(df) for df in benchmark_data.values())
print(f"\nBenchmark count: {len(benchmark_data)}")
print(f"Total sample count: {total_samples:,}")

print(f"\nBenchmark list:")
for idx, (name, df) in enumerate(sorted(benchmark_data.items(), key=lambda x: len(x[1]), reverse=True)[:], 1):
    print(f"  {idx:2d}. {name:50s} ({len(df):6,} samples)")


# ========================================
# Phase 3: Model information extraction
# ========================================

print("\n" + "=" * 80)
print("Phase 3: Model information extraction")
print("=" * 80)

# Column extraction on the first benchmark
first_benchmark = list(benchmark_data.values())[0]
all_columns = first_benchmark.columns.tolist()

print(f"\nTotal column count: {len(all_columns)}")

# Metadata column definition
metadata_columns = [
    'sample_id', 'prompt', 'query', 'question', 'input',
    'correct_answer', 'oracle_model_to_route_to',
    'oracle_model', 'eval_name', 'no_model_correct',
    'benchmark', 'difficulty', 'category', 'type'
]

# Model column extraction
potential_models = [col for col in all_columns if col not in metadata_columns]

# |model_response, |total_cost, etc. exclusion
models = []
for col in potential_models:
    # 1. |model_response included column exclusion
    if '|model_response' in col:
        continue
    # 2. |total_cost included column exclusion
    if '|total_cost' in col:
        continue
    # 3. only numerical column exclusion
    if col.isdigit():
        continue
    models.append(col)

print(f"\n{len(models)}ea model(s) extraction complete:")
for idx, model in enumerate(models, 1):
    print(f"  {idx:2d}. {model}")

# Model information save
model_info = {
    'models': models,
    'num_models': len(models),
    'metadata_columns': metadata_columns
}

with open(f'{WORK_DIR}/model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)

# ========================================
# Phase 4: Model performance matrix generation
# ========================================

print("\n" + "=" * 80)
print("Phase 4: Model performance matrix generation")
print("=" * 80)

# Performance matrix initialization
model_benchmark_performance = pd.DataFrame(
    index=models,
    columns=benchmark_data.keys(),
    dtype=float
)

print(f"\nCalculating the average performance of each model-benchmark combination...")

for model in tqdm(models, desc="Model"):
    for benchmark_name, df in benchmark_data.items():
        if model in df.columns:
            try:
                col_data = df[model]

                if col_data.dtype == 'object':
                    try:
                        col_data = pd.to_numeric(col_data, errors='coerce')
                    except:
                        model_benchmark_performance.loc[model, benchmark_name] = np.nan
                        continue

                # Average calculation (0.0 = incorrect, 1.0 = correct)
                performance = col_data.mean()

                # Save only if not NaN
                if not pd.isna(performance):
                    model_benchmark_performance.loc[model, benchmark_name] = performance
                else:
                    model_benchmark_performance.loc[model, benchmark_name] = np.nan

            except Exception as e:
                model_benchmark_performance.loc[model, benchmark_name] = np.nan
        else:
            model_benchmark_performance.loc[model, benchmark_name] = np.nan

print(f"\nPerformance matrix generation complete: {model_benchmark_performance.shape}")
print(f"  Model: {len(models)}")
print(f"  Benchmark: {len(benchmark_data)}")

# Missing value statistics
missing_count = model_benchmark_performance.isna().sum().sum()
missing_pct = (missing_count / model_benchmark_performance.size) * 100
print(f"\nMissing value information:")
print(f"  Missing value count: {missing_count}ea")
print(f"  Missing value ratio: {missing_pct:.2f}%")

# ========================================
# Phase 5: fillna strategy
# ========================================

print("\n" + "=" * 80)
print(f"Phase 5: fillna strategy application ('{FILLNA_STRATEGY}')")
print("=" * 80)

FILLNA_STRATEGY == 'keep_nan'

print("NaN strategy application complete")

# Final missing value check
missing_count_after = model_benchmark_performance.isna().sum().sum()
print(f"\nFinal missing value: {missing_count_after}ea")

# Performance matrix save
with open(f'{WORK_DIR}/model_benchmark_performance.pkl', 'wb') as f:
    pickle.dump(model_benchmark_performance, f)

with open(f'{WORK_DIR}/benchmark_data.pkl', 'wb') as f:
    pickle.dump(benchmark_data, f)

print(f"\nFile save complete:")
print(f"  {WORK_DIR}/model_benchmark_performance.pkl")
print(f"  {WORK_DIR}/benchmark_data.pkl")

# Sample print
print(f"\nPerformance matrix sample (first 5 models × 5 benchmarks)):")
sample_benchmarks = model_benchmark_performance.columns[:5]
print(model_benchmark_performance[sample_benchmarks].head().to_string())

# ========================================
# Phase 6: Test sample generation
# ========================================

print("\n" + "=" * 80)
print("Phase 6: Test sample generation")
print("=" * 80)

# Preprocessing statistics
preprocessing_stats = {
    'total_candidates': 0,
    'excluded_no_model_correct': 0,
    'excluded_few_models': 0,
    'excluded_short_query': 0,
    'excluded_invalid_predictions': 0,
    'excluded_all_wrong': 0,
    'final_samples': 0
}

# Benchmark → Task mapping
benchmark_to_task = {
    # Math
    'gsm8k': 'math_reasoning',
    'grade-school-math': 'math_reasoning',
    'math': 'math_reasoning',
    'asdiv': 'math_reasoning',
    'svamp': 'math_reasoning',
    'algebra': 'math_reasoning',

    # Code
    'humaneval': 'code_generation',
    'mbpp': 'code_generation',
    'code': 'code_generation',

    # Commonsense
    'piqa': 'commonsense_reasoning',
    'siqa': 'commonsense_reasoning',
    'hellaswag': 'commonsense_reasoning',
    'winogrande': 'commonsense_reasoning',
    'arc-challenge': 'commonsense_reasoning',
    'arc': 'commonsense_reasoning',
    'commonsense': 'commonsense_reasoning',

    # Factual QA
    'mmlu': 'factual_qa',
    'triviaqa': 'factual_qa',
    'naturalqa': 'factual_qa',

    # Conversational
    'mt-bench': 'conversational',
    'mt_bench': 'conversational',

}


def classify_task_from_benchmark(benchmark_name):
    """Benchmark Classifying tasks by name"""
    name_lower = benchmark_name.lower()

    if name_lower in benchmark_to_task:
        return benchmark_to_task[name_lower]

    for key, task in benchmark_to_task.items():
        if key in name_lower:
            return task

    return "language_understanding"

test_samples = []
excluded_samples_log = []

print(f"\nTest sample generating...")

for benchmark_name, df in tqdm(benchmark_data.items(), desc="Benchmark"):
    # Sampling up to 100 pieces from each benchmark
    sample_size = min(100, len(df))
    sampled_df = df.sample(n=sample_size, random_state=RANDOM_STATE)

    for idx, row in sampled_df.iterrows():
        preprocessing_stats['total_candidates'] += 1

        # Step 1: query quality verification
        query = row.get('query', row.get('prompt', ''))

        if len(str(query).strip()) < MIN_QUERY_LENGTH:
            preprocessing_stats['excluded_short_query'] += 1
            excluded_samples_log.append({
                'sample_id': row.get('sample_id', idx),
                'benchmark': benchmark_name,
                'reason': 'short_query',
                'query_length': len(str(query))
            })
            continue

        # Step 2: task classification ===
        task_category = classify_task_from_benchmark(benchmark_name)

        # Step 3: Model prediction collection
        model_predictions = {}
        for model in models:
            if model in row.index and not pd.isna(row[model]):
                try:
                    pred_value = float(row[model])
                    # Valid range check (acceptable only 0 or 1)
                    if pred_value in [0.0, 1.0]:
                        model_predictions[model] = pred_value
                except (ValueError, TypeError):
                    continue

        # Step 4: Check the minimum number of response models
        if len(model_predictions) < MIN_MODELS:
            preprocessing_stats['excluded_few_models'] += 1
            excluded_samples_log.append({
                'sample_id': row.get('sample_id', idx),
                'benchmark': benchmark_name,
                'reason': 'few_models',
                'num_models': len(model_predictions)
            })
            continue

        # Step 5: no_model_correct check
        if FILTER_NO_MODEL_CORRECT:
            no_model_correct = row.get('no_model_correct', False)

            if no_model_correct:
                preprocessing_stats['excluded_no_model_correct'] += 1
                excluded_samples_log.append({
                    'sample_id': row.get('sample_id', idx),
                    'benchmark': benchmark_name,
                    'reason': 'no_model_correct',
                    'no_model_correct': True
                })
                continue

        # Step 6: Check all model incorrect answers (double verification)
        all_predictions = list(model_predictions.values())
        if all(p == 0.0 for p in all_predictions):
            preprocessing_stats['excluded_all_wrong'] += 1
            excluded_samples_log.append({
                'sample_id': row.get('sample_id', idx),
                'benchmark': benchmark_name,
                'reason': 'all_models_wrong',
                'num_models': len(model_predictions)
            })
            continue

        # Final step: Add sample
        preprocessing_stats['final_samples'] += 1
        test_samples.append({
            'query': query,
            'task_category': task_category,
            'model_predictions': model_predictions,
            'benchmark': benchmark_name,
            'num_models': len(model_predictions),
            'no_model_correct': row.get('no_model_correct', False),
            'oracle_model': row.get('oracle_model_to_route_to', None)
        })

# ========================================
# Phase 7: Preprocessing statistics print
# ========================================

print("\n" + "=" * 80)
print("Phase 7: Preprocessing statistics")
print("=" * 80)

total = preprocessing_stats['total_candidates']
final = preprocessing_stats['final_samples']

print(f"\nNumber of candidate samples:              {total:,}ea")
print(f"\nReasons for exclusion:")
print(f"  Short query:                {preprocessing_stats['excluded_short_query']:,}ea ({preprocessing_stats['excluded_short_query']/total*100:.1f}%)")
print(f"  Lack of response model:           {preprocessing_stats['excluded_few_models']:,}ea ({preprocessing_stats['excluded_few_models']/total*100:.1f}%)")
print(f"  All model incorrect answers:           {preprocessing_stats['excluded_no_model_correct']:,}ea ({preprocessing_stats['excluded_no_model_correct']/total*100:.1f}%)")
print(f"  All predictive incorrect answers:           {preprocessing_stats['excluded_all_wrong']:,}ea ({preprocessing_stats['excluded_all_wrong']/total*100:.1f}%)")

total_excluded = sum([
    preprocessing_stats['excluded_short_query'],
    preprocessing_stats['excluded_few_models'],
    preprocessing_stats['excluded_no_model_correct'],
    preprocessing_stats['excluded_all_wrong']
])

print(f"\nTotal excluded:                   {total_excluded:,}ea ({total_excluded/total*100:.1f}%)")
print(f"Final sample:                 {final:,}ea ({final/total*100:.1f}%)")

# Distribution by Task
print(f"\nSample distribution by Task:")
task_counts = Counter([s['task_category'] for s in test_samples])
for task, count in sorted(task_counts.items(), key=lambda x: x[1], reverse=True):
    pct = count / len(test_samples) * 100
    print(f"  {task:30s}: {count:5d}ea ({pct:5.1f}%)")

# Sample quality statistics
num_models_per_sample = [s['num_models'] for s in test_samples]
print(f"\nSample quality:")
print(f"  Average number of response models:         {np.mean(num_models_per_sample):.2f}ea")
print(f"  Minimum number of response models:         {np.min(num_models_per_sample)}ea")
print(f"  Maximum number of response models:         {np.max(num_models_per_sample)}ea")
print(f"  Standard deviation:                  {np.std(num_models_per_sample):.2f}")

# Distribution by benchmark
print(f"\nSample distribution by benchmark (Top 10ea):")
benchmark_counts = Counter([s['benchmark'] for s in test_samples])
for benchmark, count in benchmark_counts.most_common(10):
    pct = count / len(test_samples) * 100
    print(f"  {benchmark[:40]:40s}: {count:5d}ea ({pct:5.1f}%)")

# ========================================
# Phase 8: Result save
# ========================================

print("\n" + "=" * 80)
print("Phase 8: Result save")
print("=" * 80)

# Test sample save
with open(f'{WORK_DIR}/test_samples.pkl', 'wb') as f:
    pickle.dump(test_samples, f)

# Preprocessing statistics save
with open(f'{WORK_DIR}/preprocessing_stats.json', 'w') as f:
    json.dump(preprocessing_stats, f, indent=2)

# Excluded sample log save
with open(f'{WORK_DIR}/excluded_samples_log.json', 'w') as f:
    json.dump(excluded_samples_log, f, indent=2)

# Total setting save
config = {
    'min_models': MIN_MODELS,
    'min_query_length': MIN_QUERY_LENGTH,
    'filter_no_model_correct': FILTER_NO_MODEL_CORRECT,
    'fillna_strategy': FILLNA_STRATEGY,
    'random_state': RANDOM_STATE,
    'final_samples': final,
    'total_candidates': total,
    'exclusion_rate': (1 - final/total) * 100
}

with open(f'{WORK_DIR}/preprocessing_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"\nSave complete:")
print(f"  {WORK_DIR}/test_samples.pkl ({len(test_samples):,} sample)")
print(f"  {WORK_DIR}/preprocessing_stats.json")
print(f"  {WORK_DIR}/excluded_samples_log.json ({len(excluded_samples_log):,} excluded)")
print(f"  {WORK_DIR}/preprocessing_config.json")

# ========================================
# Complete
# ========================================

print("\n" + "=" * 80)
print("Data preprocessing complete!")
print("=" * 80)

print(f"\nFinal result:")
print(f"  Total benchmark:             {len(benchmark_data)}ea")
print(f"  Total model:                 {len(models)}ea")
print(f"  Candidate sample:                 {total:,}ea")
print(f"  Fianl sample:                 {final:,}ea")
print(f"  Exclusion rate:                    {(1 - final/total)*100:.1f}%")
print(f"  fillna strategy:              {FILLNA_STRATEGY}")
print(f"  Missing Performance Matrix:          {missing_count_after}ea")