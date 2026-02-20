# ========================================
# 02.Define Variables & Functions
# ========================================

print("=" * 80)
print("Define Variables")
print("=" * 80)

# 1. cognitive_abilities define
cognitive_abilities = [
    'logical_reasoning',
    'mathematical_reasoning',
    'language_understanding',
    'common_sense',
    'code_generation',
    'factual_knowledge',
    'creative_writing',
    'multi_step_reasoning',
    'multitasking',
    'contextual_understanding',
    'domain_specific',
    'instruction_following'
]

print(f"cognitive_abilities: {len(cognitive_abilities)}ea")
for i, ability in enumerate(cognitive_abilities, 1):
    print(f"  {i:2d}. {ability}")

# 2. TASK_CATEGORIES define
TASK_CATEGORIES = {
    "math_reasoning": {
        "name": "Math Reasoning",
        "primary_abilities": ["mathematical_reasoning", "multi_step_reasoning", "logical_reasoning"]
    },
    "code_generation": {
        "name": "Code Generation",
        "primary_abilities": ["code_generation", "logical_reasoning", "instruction_following"]
    },
    "commonsense_reasoning": {
        "name": "Commonsense Reasoning",
        "primary_abilities": ["common_sense", "contextual_understanding", "language_understanding"]
    },
    "language_understanding": {
        "name": "Language Understanding",
        "primary_abilities": ["language_understanding", "contextual_understanding", "instruction_following"]
    },
    "factual_qa": {
        "name": "Factual Question Answering",
        "primary_abilities": ["factual_knowledge", "language_understanding", "domain_specific"]
    },
    "conversational": {
        "name": "Conversational",
        "primary_abilities": ["language_understanding", "contextual_understanding", "instruction_following"]
    },
    "creative_tasks": {
        "name": "Creative Tasks",
        "primary_abilities": ["creative_writing", "language_understanding", "instruction_following"]
    }
}

print("\n" + "=" * 80)
print("Define Vairables Complete!")
print("=" * 80)

# 3. NaN Safe Functions 
def safe_normalize(weights_dict):
    weights_clean = {k: (0.0 if pd.isna(v) or np.isnan(v) else v)
                     for k, v in weights_dict.items()}
    weights_clean = {k: max(0.0, v) for k, v in weights_clean.items()}

    total = sum(weights_clean.values())
    if total > 0:
        return {k: v / total for k, v in weights_clean.items()}
    else:
        n = len(weights_clean)
        return {k: 1.0 / n for k in weights_clean.keys()}

class AdaptiveBenchmarkWeightingSafe:
    def __init__(self, model_benchmark_performance, models, temperature=1.0):
        self.model_benchmark_performance = model_benchmark_performance
        self.models = models
        self.temperature = temperature
        self.task_weights = {}

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

        if scores_array.max() > scores_array.min():
            normalized = (scores_array - scores_array.min()) / (scores_array.max() - scores_array.min())
        else:
            normalized = np.ones_like(scores_array) * 0.5

        scaled_scores = normalized / self.temperature
        weights_array = softmax(scaled_scores)

        weights = {model: float(w) for model, w in zip(self.models, weights_array)}
        self.task_weights[benchmark_name] = weights

        return weights

    def get_weights_safe(self, task_category):
        return self.calculate_weights(task_category)

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

def weighted_voting_safe(model_predictions, weights):
    weighted_sum = 0.0
    total_weight = 0.0

    for model, pred in model_predictions.items():
        if model in weights:
            weight = weights[model]

            if pd.isna(weight) or np.isnan(weight):
                weight = 0.0
            if pd.isna(pred) or np.isnan(pred):
                pred = 0.0

            weighted_sum += weight * pred
            total_weight += weight

    if total_weight > 0:
        weighted_avg = weighted_sum / total_weight
    else:
        weighted_avg = 0.5

    return 1 if weighted_avg >= 0.5 else 0

def select_top_k_models_safe(weights, k1_threshold, k2_threshold):
    weights_clean = {k: (0.0 if pd.isna(v) or np.isnan(v) else v)
                     for k, v in weights.items()}

    max_weight = max(weights_clean.values())

    if max_weight > k1_threshold:
        K = 1
    elif max_weight > k2_threshold:
        K = 2
    else:
        K = 3

    sorted_models = sorted(weights_clean.items(), key=lambda x: x[1], reverse=True)
    top_k_models = dict(sorted_models[:K])

    return K, safe_normalize(top_k_models)

def estimate_ability_scores_safe(model_performance, benchmark_ability_matrix, alpha=1.0):
    common_benchmarks = list(set(model_performance.columns) &
                             set(benchmark_ability_matrix.index))

    if len(common_benchmarks) == 0:
        print("No common benchmarks")
        return pd.DataFrame()

    num_abilities = benchmark_ability_matrix.shape[1]
    ability_scores = pd.DataFrame(
        index=model_performance.index,
        columns=benchmark_ability_matrix.columns,
        dtype=float
    )

    X = benchmark_ability_matrix.loc[common_benchmarks].values

    print(f"  Common Benchmarks: {len(common_benchmarks)}ea")
    print(f"  Number of Capability Dimensions: {X.shape[1]}ea")

    for model in tqdm(model_performance.index, desc="  Model-specific capability estimation"):
        y = model_performance.loc[model, common_benchmarks].values

        valid_idx = ~pd.isna(y)

        if valid_idx.sum() < 5:
            ability_scores.loc[model] = 0.5
            continue

        X_valid = X[valid_idx]
        y_valid = y[valid_idx]

        ridge = Ridge(alpha=alpha, fit_intercept=True, random_state=42)
        ridge.fit(X_valid, y_valid)

        scores = ridge.coef_
        scores = np.clip(scores, 0.0, 1.0)

        ability_scores.loc[model] = scores

    return ability_scores

print("Define NaN Safe Functions Complete!")