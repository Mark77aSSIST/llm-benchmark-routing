# ========================================
# 04. Evaluation
# ========================================

# Phase 1: Evaluation Preparation
# ========================================
print("\n1. K-Threshold Calculating...")

# Load from k_thresholds.json or use default values
try:
    import json
    with open(f'{WORK_DIR}/k_thresholds.json', 'r') as f:
        k_thresh = json.load(f)
    K1_THRESHOLD = k_thresh['k1_threshold']
    K2_THRESHOLD = k_thresh['k2_threshold']
    print(f"Load from JSON: K1={K1_THRESHOLD:.4f}, K2={K2_THRESHOLD:.4f}")
except:
    K1_THRESHOLD = 0.09
    K2_THRESHOLD = 0.09
    print(f"Use Default Values: K1={K1_THRESHOLD:.4f}, K2={K2_THRESHOLD:.4f}")

# Evaluation Method Definition
def evaluate_method(method_name, weight_calculator, test_samples, use_topk=False):
    results = {
        'correct': 0,
        'total': 0,
        'k_distribution': {1: 0, 2: 0, 3: 0},
        'predictions': [],
        'avg_calls': 0.0
    }

    total_calls = 0

    for sample in tqdm(test_samples, desc=f"  {method_name}"):
        benchmark = sample['benchmark']
        model_predictions = sample['model_predictions']

        # Ground truth
        oracle_model = sample.get('oracle_model', None)
        if oracle_model and oracle_model in model_predictions:
            ground_truth = model_predictions[oracle_model]
        else:
            votes = list(model_predictions.values())
            ground_truth = 1 if sum(votes) >= len(votes)/2 else 0

        # Calculate weights with benchmarks
        weights = weight_calculator.calculate_weights(benchmark)

        # Top-K Selection
        if use_topk:
            K, selected_weights = select_top_k_models_safe(weights, K1_THRESHOLD, K2_THRESHOLD)
            results['k_distribution'][K] += 1
            total_calls += K
        else:
            selected_weights = weights
            total_calls += len(models)

        # Weighted Voting
        prediction = weighted_voting_safe(model_predictions, selected_weights)

        results['total'] += 1
        if prediction == ground_truth:
            results['correct'] += 1

        results['predictions'].append(prediction)

    results['accuracy'] = results['correct'] / results['total'] if results['total'] > 0 else 0.0
    results['avg_calls'] = total_calls / results['total'] if results['total'] > 0 else 0.0

    return results

print("Evaluate Method Definition Complete")

# Oracle Calculation
print("\n3. Oracle Calculating...")
oracle_results = {
    'correct': 0,
    'total': 0
}

for sample in tqdm(test_samples, desc="  Oracle"):
    oracle_model = sample.get('oracle_model', None)

    if oracle_model and oracle_model in sample['model_predictions']:
        oracle_pred = sample['model_predictions'][oracle_model]
        oracle_results['correct'] += 1
    else:
        oracle_results['correct'] += 1

    oracle_results['total'] += 1

oracle_results['accuracy'] = oracle_results['correct'] / oracle_results['total']

print(f"Oracle Accuracy: {oracle_results['accuracy']:.4f}")

# Best Single Model Calculation
print("\n4. Best Single Model Calculating...")

best_single_results = {
    'correct': 0,
    'total': 0,
    'avg_calls': 1.0
}

task_best_models = {}   # Select the best model for each task
for task_id in TASK_CATEGORIES.keys():
    weights = abw.calculate_weights(task_id)
    best_model = max(weights.items(), key=lambda x: x[1])[0]
    task_best_models[task_id] = best_model

for sample in tqdm(test_samples, desc="  Best Single"):
    task = sample['task_category']
    best_model = task_best_models.get(task, models[0])

    if best_model in sample['model_predictions']:
        prediction = sample['model_predictions'][best_model]
    else:
        prediction = 0

    # Ground truth
    oracle_model = sample.get('oracle_model', None)
    if oracle_model and oracle_model in sample['model_predictions']:
        ground_truth = sample['model_predictions'][oracle_model]
    else:
        votes = list(sample['model_predictions'].values())
        ground_truth = 1 if sum(votes) >= len(votes)/2 else 0

    best_single_results['total'] += 1
    if prediction == ground_truth:
        best_single_results['correct'] += 1

best_single_results['accuracy'] = best_single_results['correct'] / best_single_results['total']

print(f"Best Single Accuracy: {best_single_results['accuracy']:.4f}")
print()
print("=" * 80)
print("Evaluation Preparation Complete")
print("=" * 80)
print()
print("Definied Variables:")
print(f"  - K1_THRESHOLD: {K1_THRESHOLD:.4f}")
print(f"  - K2_THRESHOLD: {K2_THRESHOLD:.4f}")
print(f"  - oracle_results: {oracle_results['accuracy']:.2%}")
print(f"  - best_single_results: {best_single_results['accuracy']:.2%}")

# Phase 2: Evaluation
# ========================================
print("\n" + "=" * 80)
print("Evaluation Start")
print("=" * 80)

# Uniform
print("\nEvaluating Uniform Model...")
uniform_results = evaluate_uniform(test_samples)

# ABW
print("\nEvaluating ABW Model...")
abw_results = evaluate_method("ABW", abw, test_samples, use_topk=False)

# SPW
print("\nEvaluating SPW Model...")
spw_results = evaluate_method("SPW", spw, test_samples, use_topk=False)

# Hybrid
print("\nEvaluating Hybrid Model...")
hybrid_results = evaluate_method("Hybrid", hybrid, test_samples, use_topk=False)

# Top-K
print("\nEvaluating Top-K Model...")
topk_results = evaluate_method("Top-K", hybrid, test_samples, use_topk=True)

# Comparing Routing Results
print("\n" + "=" * 80)
print("Comparing Routing Results")
print("=" * 80)

print(f"\n{'Method':<15s} {'Accuracy':>10s} {'Avg Calls':>12s}")
print("-" * 60)

for method_name in ['uniform', 'abw', 'spw', 'hybrid', 'topk']:
    if method_name == 'uniform':
        result = uniform_results
    elif method_name == 'abw':
        result = abw_results
    elif method_name == 'spw':
        result = spw_results
    elif method_name == 'hybrid':
        result = hybrid_results
    elif method_name == 'topk':
        result = topk_results

    acc = result['accuracy']
    calls = result['avg_calls']

    print(f"{method_name.upper():<15s} {acc:>9.2%} {calls:>11.2f}")


# Statistics Significance
print("\n" + "=" * 80)
print("Test for statistical significance")
print("=" * 80)

# Uniform vs ABW
uniform_preds = np.array(uniform_results['predictions'])
abw_preds = np.array(abw_results['predictions'])
spw_preds = np.array(spw_results['predictions'])
hybrid_preds = np.array(hybrid_results['predictions'])
topk_preds = np.array(topk_results['predictions'])

# ABW vs Uniform
if not np.array_equal(abw_preds, uniform_preds):
    t_stat, p_value = stats.ttest_rel(abw_preds, uniform_preds)
    print(f"\nABW vs Uniform:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Significant (p<0.05): {'Yes ✓' if p_value < 0.05 else 'No ✗'}")
else:
    print(f"\nABW vs Uniform: the same prediction")

# Hybrid vs Uniform
if not np.array_equal(hybrid_preds, uniform_preds):
    t_stat, p_value = stats.ttest_rel(hybrid_preds, uniform_preds)
    print(f"\nHybrid vs Uniform:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Significant (p<0.05): {'Yes ✓' if p_value < 0.05 else 'No ✗'}")
else:
    print(f"\nHybrid vs Uniform: the same prediction")

# Top-K vs Hybrid
if not np.array_equal(topk_preds, hybrid_preds):
    t_stat, p_value = stats.ttest_rel(topk_preds, hybrid_preds)
    print(f"\nTop-K vs Hybrid:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Significant (p<0.05): {'Yes ✓' if p_value < 0.05 else 'No ✗'}")
else:
    print(f"\nTop-K vs Hybrid: the same prediction")

# Final Results Save
all_results = {
    'oracle': oracle_results,
    'best_single': best_single_results,
    'uniform': uniform_results,
    'abw': abw_results,
    'spw': spw_results,
    'hybrid': hybrid_results,
    'topk': topk_results
}

import pickle
with open(f'{WORK_DIR}/evaluation_results_FINAL.pkl', 'wb') as f:
    pickle.dump(all_results, f)

print("\nFinal Results Save: evaluation_results_FINAL.pkl")

print("\n" + "=" * 80)
print("Evaluation Complete!")
print("=" * 80)