import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Fetch data
def fetch_data():
    data = yf.download("ES=F", start="2020-01-01", end="2023-01-01", interval="1h")
    data = data.dropna()
    return data

# Data Preprocessing
def preprocess_data(data):
    # Normalize
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)
    data = pd.DataFrame(data_normalized, columns=data.columns)

    # Outlier detection
    z_scores = np.abs(zscore(data))
    data = data[(z_scores < 3).all(axis=1)]
    
    return data

# Feature Engineering
def feature_engineering(data):
    # Technical Indicators
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_10'] = data['Close'].rolling(window=10).mean()

    # Time Features
    data['Day_of_Week'] = data.index.dayofweek
    data['Month'] = data.index.month

    # Rolling Statistics
    data['Rolling_Mean'] = data['Close'].rolling(window=5).mean()
    data['Rolling_Std'] = data['Close'].rolling(window=5).std()

    # Exponential Moving Average
    data['EMA_5'] = data['Close'].ewm(span=5, adjust=False).mean()
    data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()

    # Drop NaN values introduced by rolling computations
    data = data.dropna()
    
    return data

# Random Bayesian Forest Model Code

class Data:
    def __init__(self, features, prices):
        self.features = features
        self.prices = prices

class TreeNode:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, gaussian_params=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gaussian_params = gaussian_params

class LeafNode:
    def __init__(self, gaussian_params):
        self.gaussian_params = gaussian_params

def compute_likelihood(data, mean, variance):
    epsilon = 1e-8
    return np.exp(-((data.prices - mean)**2) / (2 * (variance + epsilon))) / np.sqrt(2 * np.pi * (variance + epsilon))

def bayesian_update(prior_mean, prior_variance, likelihood_mean, likelihood_variance, data_size):
    precision_prior = 1 / prior_variance
    precision_likelihood = data_size / likelihood_variance
    posterior_variance = 1 / (precision_prior + precision_likelihood)
    posterior_mean = (prior_mean / prior_variance + likelihood_mean * data_size / likelihood_variance) * posterior_variance
    return posterior_mean, posterior_variance

def hypothetical_split(data, feature_idx, threshold):
    left_mask = data.features[:, feature_idx] < threshold
    right_mask = ~left_mask
    left_data = Data(data.features[left_mask], data.prices[left_mask])
    right_data = Data(data.features[right_mask], data.prices[right_mask])
    return left_data, right_data

def bic_or_expected_utility(data, mean, variance):
    n = len(data.prices)
    log_likelihood = np.sum(np.log(compute_likelihood(data, mean, variance)))
    return log_likelihood - 1/2 * np.log(n)

def select_best_split(data, max_features=None):
    best_score = float('-inf')
    best_split = None
    n, m = data.features.shape
    if max_features is None:
        features_to_check = range(m)
    else:
        features_to_check = np.random.choice(m, max_features, replace=False)
    for feature_idx in features_to_check:
        thresholds = np.unique(data.features[:, feature_idx])
        for threshold in thresholds:
            left_data, right_data = hypothetical_split(data, feature_idx, threshold)
            if len(left_data.prices) == 0 or len(right_data.prices) == 0:
                continue
            prior_mean, prior_variance = 0, 1
            left_likelihood_mean, left_likelihood_variance = np.mean(left_data.prices), np.var(left_data.prices)
            right_likelihood_mean, right_likelihood_variance = np.mean(right_data.prices), np.var(right_data.prices)
            left_post_mean, left_post_var = bayesian_update(prior_mean, prior_variance, left_likelihood_mean, left_likelihood_variance, len(left_data.prices))
            right_post_mean, right_post_var = bayesian_update(prior_mean, prior_variance, right_likelihood_mean, right_likelihood_variance, len(right_data.prices))
            score = bic_or_expected_utility(left_data, left_post_mean, left_post_var) + bic_or_expected_utility(right_data, right_post_mean, right_post_var)
            if score > best_score:
                best_score = score
                best_split = (feature_idx, threshold)
    return best_split

def bayesian_decision_tree(data, max_depth, max_features=None):
    if max_depth == 0:
        likelihood_mean, likelihood_variance = np.mean(data.prices), np.var(data.prices)
        prior_mean, prior_variance = 0, 1
        post_mean, post_var = bayesian_update(prior_mean, prior_variance, likelihood_mean, likelihood_variance, len(data.prices))
        return LeafNode((post_mean, post_var))
    best_split = select_best_split(data, max_features)
    if best_split is None:
        likelihood_mean, likelihood_variance = np.mean(data.prices), np.var(data.prices)
        prior_mean, prior_variance = 0, 1
        post_mean, post_var = bayesian_update(prior_mean, prior_variance, likelihood_mean, likelihood_variance, len(data.prices))
        return LeafNode((post_mean, post_var))
    feature_idx, threshold = best_split
    left_data, right_data = hypothetical_split(data, feature_idx, threshold)
    left_child = bayesian_decision_tree(left_data, max_depth - 1, max_features)
    right_child = bayesian_decision_tree(right_data, max_depth - 1, max_features)
    return TreeNode(feature_idx, threshold, left_child, right_child)

def bootstrap(data, size=None):
    if size is None:
        size = len(data.prices)
    indices = np.random.choice(len(data.prices), size, replace=True)
    return Data(data.features[indices], data.prices[indices])

def random_bayesian_forest(data, n_trees, max_depth, max_features=None):
    trees = []
    for _ in range(n_trees):
        bootstrapped_data = bootstrap(data)
        tree = bayesian_decision_tree(bootstrapped_data, max_depth, max_features)
        trees.append(tree)
    return trees

def traverse_tree(node, sample):
    if isinstance(node, LeafNode):
        mean, variance = node.gaussian_params
        return np.random.normal(mean, np.sqrt(variance))
    feature_idx, threshold = node.feature_idx, node.threshold
    if sample[feature_idx] < threshold:
        return traverse_tree(node.left, sample)
    else:
        return traverse_tree(node.right, sample)

def predict_forest(forest, sample):
    predictions = [traverse_tree(tree, sample) for tree in forest]
    return np.mean(predictions)

def evaluate_model(forest, test_data):
    predictions = [predict_forest(forest, sample) for sample in test_data.features]
    rmse = np.sqrt(mean_squared_error(test_data.prices, predictions))
    mae = mean_absolute_error(test_data.prices, predictions)
    return rmse, mae

# Fetch real data
data = fetch_data()

# Preprocess the data
data = preprocess_data(data)

# Feature Engineering
data = feature_engineering(data)

# Train-Test Split
X = data.drop(columns=['Close']).values
y = data['Close'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
train_data = Data(X_train, y_train)
test_data = Data(X_test, y_test)

# Model Training
forest = random_bayesian_forest(train_data, n_trees=10, max_depth=3)

# Model Evaluation
rmse, mae = evaluate_model(forest, test_data)

print(f"RMSE: {rmse}, MAE: {mae}")
