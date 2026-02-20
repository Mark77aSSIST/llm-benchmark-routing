# ========================================
# 03.Model Definition
# ========================================

# Phase 1: Uniform Model
# ========================================
print("\n" + "=" * 80)
print("Phase 1: Uniform")
print("=" * 80)

def evaluate_uniform(test_samples):
    results = {
        'correct': 0,
        'total': 0,
        'k_distribution': {1: 0, 2: 0, 3: 0},
        'predictions': [],
        'avg_calls': 11.0
    }

    uniform_weights = {m: 1.0/len(models) for m in models}

    for sample in tqdm(test_samples, desc="  Uniform"):
        model_predictions = sample['model_predictions']

        # Ground truth
        oracle_model = sample.get('oracle_model', None)
        if oracle_model and oracle_model in model_predictions:
            ground_truth = model_predictions[oracle_model]
        else:
            votes = list(model_predictions.values())
            ground_truth = 1 if sum(votes) >= len(votes)/2 else 0

        # Using Equal Weight
        prediction = weighted_voting_safe(model_predictions, uniform_weights)

        results['total'] += 1
        if prediction == ground_truth:
            results['correct'] += 1

        results['predictions'].append(prediction)

    results['accuracy'] = results['correct'] / results['total'] if results['total'] > 0 else 0.0

    return results

print("Uniform Model Definition Complete")

# Phase 2: ABW (Adaptive Benchmark Weighting)
# ========================================
print("\n" + "=" * 80)
print("Phase 1: ABW (Adaptive Benchmark Weighting)")
print("=" * 80)

class AdaptiveBenchmarkWeighting:
    def __init__(self, model_benchmark_performance, models, temperature=1.0):
        self.model_benchmark_performance = model_benchmark_performance
        self.models = models
        self.temperature = temperature
        self.task_weights = {}
        self.benchmark_stats = {}

    def calculate_weights(self, benchmark_name):
        if benchmark_name in self.task_weights:
            return self.task_weights[benchmark_name]

        if benchmark_name not in self.model_benchmark_performance.columns:
            uniform_weight = 1.0 / len(self.models)
            return {model: uniform_weight for model in self.models}

        scores = {}
        for model in self.models:
            if model in self.model_benchmark_performance.index:
                score = self.model_benchmark_performance.loc[model, benchmark_name]
                if pd.isna(score) or np.isnan(score):
                    scores[model] = 0.0
                else:
                    scores[model] = float(score)
            else:
                scores[model] = 0.0

        scores_array = np.array(list(scores.values()))

        # Min-Max normalization
        if scores_array.max() > scores_array.min():
            normalized = (scores_array - scores_array.min()) / (scores_array.max() - scores_array.min())
        else:
            normalized = np.ones_like(scores_array) * 0.5

        # Softmax
        scaled_scores = normalized / self.temperature
        weights_array = softmax(scaled_scores)

        weights = {model: float(w) for model, w in zip(self.models, weights_array)}

        # Statistics save
        self.benchmark_stats[benchmark_name] = {
            'max': float(weights_array.max()),
            'min': float(weights_array.min()),
            'std': float(weights_array.std()),
            'avg_accuracy': float(scores_array.mean())
        }

        self.task_weights[benchmark_name] = weights
        return weights

    def get_weights(self, task_category):
        return self.calculate_weights(task_category)

# ABW initialization (Temperature=1.0 Default value)
print("\nABW initializing...")
abw = AdaptiveBenchmarkWeighting(
    model_benchmark_performance,
    models,
    temperature=1.0
)

# Calculate weights for all benchmarks
print("\nCalculating weights for all benchmarks...")
for benchmark_name in tqdm(benchmark_data.keys(), desc="Benchmarks"):
    abw.calculate_weights(benchmark_name)

print(f"  Number of Benchmarks Calculated: {len(abw.task_weights)}")

# Sample Weights Print
sample_benchmark = list(benchmark_data.keys())[0]
sample_weights = abw.get_weights(sample_benchmark)
print(f"\nSample Benchmarks ({sample_benchmark}) Weights:")
for model, weight in sorted(sample_weights.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {model:40s}: {weight:.4f}")

# ABW save
with open(f'{WORK_DIR}/abw_weights.pkl', 'wb') as f:
    pickle.dump(abw.task_weights, f)

# Expanding the Temperature Range
temperature_candidates = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

print(f"\nTemperature Candidates: {temperature_candidates}")

# Preparing data by task
task_data = {}

for task_id in TASK_CATEGORIES.keys():
    task_samples_list = [s for s in test_samples if s['task_category'] == task_id]

    if len(task_samples_list) >= 20:
        X_predictions = []
        for sample in task_samples_list:
            model_preds = [float(sample['model_predictions'].get(model, 0.0)) for model in models]
            X_predictions.append(model_preds)

        task_data[task_id] = np.array(X_predictions)

print(f"{len(task_data)}ea task data ready")

# Temperature Tuning
print("\nTemperature Tuning...")

best_temp = 1.0
best_accuracy = 0.0
temp_results = []

for temp in temperature_candidates:
    abw_temp = AdaptiveBenchmarkWeighting(
        model_benchmark_performance,
        models,
        temperature=temp
    )

    total_correct = 0
    total_samples = 0

    # Calculate weight variance
    sample_task = list(task_data.keys())[0]
    weights = abw_temp.calculate_weights(sample_task)
    weights_std = np.std(list(weights.values()))

    for task_id, X in task_data.items():
        weights = abw_temp.calculate_weights(task_id)
        weights_array = np.array([weights.get(m, 0.0) for m in models])

        weighted_votes = (X @ weights_array).reshape(-1)
        predictions = (weighted_votes >= 0.5).astype(int)

        ground_truth = (X.mean(axis=1) >= 0.5).astype(int)

        total_correct += (predictions == ground_truth).sum()
        total_samples += len(predictions)

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    temp_results.append({
        'temp': temp,
        'accuracy': accuracy,
        'weights_std': weights_std
    })

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_temp = temp

print(f"\nOptimum Temperature: {best_temp:.2f}")
print(f"   Accuracy: {best_accuracy:.4f}")

# Regenerate ABW with optimum temperature
abw = AdaptiveBenchmarkWeighting(model_benchmark_performance, models, temperature=best_temp)

for benchmark_name in benchmark_data.keys():
    abw.calculate_weights(benchmark_name)

print("ABW Model Definition Complete")

# Phase 3: SPW (Static Profile Weighting)
# ========================================
class StaticProfileWeightingSafe:
    def __init__(self, model_profiles, task_categories, temperature=1.0):
        self.model_profiles = model_profiles
        self.task_categories = task_categories
        self.temperature = temperature
        self.models = list(model_profiles.keys())

    def calculate_weights(self, task_category):
        if task_category not in self.task_categories:
            return {m: 1.0/len(self.models) for m in self.models}

        primary_abilities = self.task_categories[task_category]['primary_abilities']
        model_scores = {}

        for model in self.models:
            profile = self.model_profiles[model]
            abilities = profile.get('cognitive_abilities', {})

            relevant_scores = []
            for ability in primary_abilities:
                score = abilities.get(ability, None)
                if score is not None and not pd.isna(score) and not np.isnan(score):
                    relevant_scores.append(float(score))

            if len(relevant_scores) > 0:
                model_scores[model] = np.mean(relevant_scores)
            else:
                model_scores[model] = 0.5

        scores_array = np.array(list(model_scores.values()))
        scaled_scores = scores_array / self.temperature
        scaled_scores = scaled_scores - np.max(scaled_scores)
        exp_scores = np.exp(scaled_scores)
        weights_array = exp_scores / exp_scores.sum()

        return {m: float(w) for m, w in zip(self.models, weights_array)}

def evaluate_spw(test_samples):
    results = {
        'correct': 0,
        'total': 0,
        'k_distribution': {1: 0, 2: 0, 3: 0},
        'predictions': [],
        'avg_calls': 11.0
    }

    for sample in tqdm(test_samples, desc="  SPW (Task)"):
        task = sample['task_category']
        model_predictions = sample['model_predictions']

        # Ground truth
        oracle_model = sample.get('oracle_model', None)
        if oracle_model and oracle_model in model_predictions:
            ground_truth = model_predictions[oracle_model]
        else:
            votes = list(model_predictions.values())
            ground_truth = 1 if sum(votes) >= len(votes)/2 else 0

        # SPW weights
        weights_spw = spw.calculate_weights(task)

        # Weighted voting
        prediction = weighted_voting_safe(model_predictions, weights_spw)

        results['total'] += 1
        if prediction == ground_truth:
            results['correct'] += 1

        results['predictions'].append(prediction)

    results['accuracy'] = results['correct'] / results['total'] if results['total'] > 0 else 0.0

    return results

# Benchmatk-Ability Mapping
def map_benchmark_to_abilities_improved(benchmark_name):
    abilities = []
    name_lower = benchmark_name.lower()

    # Math
    if any(x in name_lower for x in ['math', 'gsm', 'asdiv', 'svamp', 'algebra']):
        abilities.extend([
            ('mathematical_reasoning', 0.90),
            ('multi_step_reasoning', 0.70),
            ('logical_reasoning', 0.60)
        ])

    # Code
    if any(x in name_lower for x in ['code', 'humaneval', 'mbpp', 'programming']):
        abilities.extend([
            ('code_generation', 0.95),
            ('logical_reasoning', 0.80),
            ('instruction_following', 0.70)
        ])

    # Commonsense
    if any(x in name_lower for x in ['hellaswag', 'winogrande', 'piqa', 'siqa', 'arc']):
        abilities.extend([
            ('common_sense', 0.90),
            ('contextual_understanding', 0.70),
            ('language_understanding', 0.60)
        ])

    # Factual QA
    if any(x in name_lower for x in ['mmlu', 'trivia', 'natural', 'fact']):
        abilities.extend([
            ('factual_knowledge', 0.95),
            ('domain_specific', 0.75),
            ('language_understanding', 0.85)
        ])

    # Conversational
    if any(x in name_lower for x in ['mt-bench', 'mt_bench', 'conversation', 'dialog']):
        abilities.extend([
            ('language_understanding', 0.90),
            ('contextual_understanding', 0.80),
            ('instruction_following', 0.70)
        ])

    if not abilities:
        abilities = [('language_understanding', 0.60)] 

    return abilities


# Benchmark-Ability Matrix Generation
print("\nBenchmark-Ability Matrix Generating...")

benchmark_ability_matrix = pd.DataFrame(
    0.0,
    index=list(benchmark_data.keys()),
    columns=cognitive_abilities
)

for benchmark_name in benchmark_data.keys():
    abilities_map = map_benchmark_to_abilities_improved(benchmark_name)
    for ability, weight in abilities_map:
        if ability in benchmark_ability_matrix.columns:
            benchmark_ability_matrix.loc[benchmark_name, ability] = weight

# Model ability estimation
print("\nEstimating model ability (Ridge alpha=0.1)...")

model_ability_scores = estimate_ability_scores_safe(
    model_benchmark_performance,
    benchmark_ability_matrix,
    alpha=0.1
)

print(f"Model ability estimation completed")

# Model Profile Generation
model_profiles = {}
for model in models:
    if model in model_ability_scores.index:
        abilities_dict = model_ability_scores.loc[model].to_dict()
        model_profiles[model] = {
            'cognitive_abilities': abilities_dict,
            'overall_score': np.mean([v for v in abilities_dict.values() if not pd.isna(v)])
        }

# SPW initialization
spw = StaticProfileWeightingSafe(model_profiles, TASK_CATEGORIES, temperature=1.0)

print("SPW Model Definition Complete")

# Phase 4: Hybrid (SPW+ABW)
# ========================================
class HybridWeightingSafe:
    def __init__(self, spw, abw, alpha=0.5):
        self.spw = spw
        self.abw = abw
        self.alpha = alpha
        self.models = spw.models

    def calculate_weights(self, task_category):
        spw_weights = self.spw.calculate_weights(task_category)
        abw_weights = self.abw.calculate_weights(task_category)

        hybrid_weights = {}
        for model in self.models:
            spw_w = spw_weights.get(model, 0.0)
            abw_w = abw_weights.get(model, 0.0)

            if pd.isna(spw_w) or np.isnan(spw_w):
                spw_w = 0.0
            if pd.isna(abw_w) or np.isnan(abw_w):
                abw_w = 0.0

            hybrid_w = self.alpha * spw_w + (1 - self.alpha) * abw_w
            hybrid_weights[model] = hybrid_w

        return safe_normalize(hybrid_weights)

def evaluate_hybrid(test_samples):
    results = {
        'correct': 0,
        'total': 0,
        'k_distribution': {1: 0, 2: 0, 3: 0},
        'predictions': [],
        'avg_calls': 11.0
    }

    for sample in tqdm(test_samples, desc="  Hybrid"):
        task = sample['task_category']
        benchmark = sample['benchmark']
        model_predictions = sample['model_predictions']

        # Ground truth
        oracle_model = sample.get('oracle_model', None)
        if oracle_model and oracle_model in model_predictions:
            ground_truth = model_predictions[oracle_model]
        else:
            votes = list(model_predictions.values())
            ground_truth = 1 if sum(votes) >= len(votes)/2 else 0

        # ✅ Hybrid Weights (SPW: task, ABW: benchmark)
        spw_weights = spw.calculate_weights(task)
        abw_weights = pbw.calculate_weights(benchmark)

        # Hybrid Combination (alpha=0.5)
        hybrid_weights = {}
        for model in models:
            spw_w = spw_weights.get(model, 0.0)
            abw_w = abw_weights.get(model, 0.0)

            # NaN Check
            if pd.isna(spw_w) or np.isnan(spw_w):
                spw_w = 0.0
            if pd.isna(abw_w) or np.isnan(abw_w):
                abw_w = 0.0

            hybrid_weights[model] = 0.5 * spw_w + 0.5 * abw_w

        # Normalization
        total_weight = sum(hybrid_weights.values())
        if total_weight > 0:
            hybrid_weights = {m: w/total_weight for m, w in hybrid_weights.items()}
        else:
            hybrid_weights = {m: 1.0/len(models) for m in models}

        # Weighted Voting
        prediction = weighted_voting_safe(model_predictions, hybrid_weights)

        results['total'] += 1
        if prediction == ground_truth:
            results['correct'] += 1

        results['predictions'].append(prediction)

    results['accuracy'] = results['correct'] / results['total'] if results['total'] > 0 else 0.0

    return results

# Hybrid initialization
print("\nHybrid Weighting initializing (alpha=0.5)...")
hybrid = HybridWeightingSafe(spw, abw, alpha=0.5)

print("Hybrid Model Definition Complete")

# Phase 4: Hybrid (SPW+ABW)
# ========================================
def evaluate_topk(test_samples):
    results = {
        'correct': 0,
        'total': 0,
        'k_distribution': {1: 0, 2: 0, 3: 0},
        'predictions': [],
        'avg_calls': 0.0
    }

    total_calls = 0

    for sample in tqdm(test_samples, desc="  Top-K"):
        task = sample['task_category']
        benchmark = sample['benchmark']
        model_predictions = sample['model_predictions']

        # Ground truth
        oracle_model = sample.get('oracle_model', None)
        if oracle_model and oracle_model in model_predictions:
            ground_truth = model_predictions[oracle_model]
        else:
            votes = list(model_predictions.values())
            ground_truth = 1 if sum(votes) >= len(votes)/2 else 0

        # Hybrid Weights
        spw_weights = spw.calculate_weights(task)
        abw_weights = pbw.calculate_weights(benchmark)

        hybrid_weights = {}
        for model in models:
            spw_w = spw_weights.get(model, 0.0)
            abw_w = abw_weights.get(model, 0.0)

            if pd.isna(spw_w) or np.isnan(spw_w):
                spw_w = 0.0
            if pd.isna(abw_w) or np.isnan(abw_w):
                abw_w = 0.0

            hybrid_weights[model] = 0.5 * spw_w + 0.5 * abw_w

        # Normalization
        total_weight = sum(hybrid_weights.values())
        if total_weight > 0:
            hybrid_weights = {m: w/total_weight for m, w in hybrid_weights.items()}
        else:
            hybrid_weights = {m: 1.0/len(models) for m in models}

        # Top-K Selection
        K, selected_weights = select_top_k_models_safe(hybrid_weights, K1_THRESHOLD, K2_THRESHOLD)
        results['k_distribution'][K] += 1
        total_calls += K

        # Weighted Voting
        prediction = weighted_voting_safe(model_predictions, selected_weights)

        results['total'] += 1
        if prediction == ground_truth:
            results['correct'] += 1

        results['predictions'].append(prediction)

    results['accuracy'] = results['correct'] / results['total'] if results['total'] > 0 else 0.0
    results['avg_calls'] = total_calls / results['total'] if results['total'] > 0 else 0.0

    return results

print("Top-K Model Definition Complete")