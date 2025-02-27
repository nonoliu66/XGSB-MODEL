import rasterio
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix
from tqdm import tqdm  # Used for progress indication
from deap import base, creator, tools, algorithms

# Read a CSV file of sample points (with features and labels)
samples = pd.read_csv(r"C:\Users\liu\Downloads\new_labeled_data.csv")

# Extract features and labels
X = samples.drop(['landcover'], axis=1)
y = samples['landcover']

# Dataset segmentation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train an XGBoost model
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X_train.columns.tolist())
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=X_train.columns.tolist())
print(y_test)
def evaluate(individual):
    # Map the parameter values in the individual to the actual value range
    num_boost_round = int(individual[0] * 1000)
    max_depth = max(int(individual[1] * 10), 1)
    reg_alpha = np.clip(individual[2], 0, None)
    min_child_weight = max(0, individual[3])
    colsample_bynode = max(0, individual[4])
    subsample = max(0, individual[5])
    learning_rate = max(0, individual[6])
    scale_pos_weight = max(0, individual[7])
    # Create an XGBoost classifier
    params = {
        'objective': 'multi:softprob',
        'booster': 'gbtree',
        'num_class': 5,
        'max_depth': max_depth,
        'reg_alpha': reg_alpha,
        'min_child_weight': min_child_weight,
        'colsample_bynode': colsample_bynode,
        'subsample': subsample,
        'learning_rate': learning_rate,
    }
    # Train the model
    bst = xgb.train(params, dtrain, num_boost_round=num_boost_round,evals=[(dtest, 'eval'), (dtrain, 'train')],
                    early_stopping_rounds=10)
    y_pred_prob = bst.predict(dtest)
    # Evaluate the model
    y_pred = np.argmax(y_pred_prob, axis=1)
    # Calculate each indicator as fitness
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
    f1 = f1_score(y_test, y_pred, average='macro')
    return accuracy, precision, recall, auc, f1


# Initialize GA parameters
creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0, 1.0, 1.0, 1.0))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("attribute", np.random.uniform, low=0, high=0.5, size=8)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attribute)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Initialize the population
population = toolbox.population(n=50)

# Run the genetic algorithm
NGEN = 20  # 迭代次数
for gen in range(NGEN):
    offspring = algorithms.varOr(population, toolbox, lambda_=len(population), cxpb=0.5, mutpb=0.1)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

# Output the optimal combination of parameters and indicator values
best_individual = tools.selBest(population, k=1)[0]
best_metrics = best_individual.fitness.values
print("Best individual:", best_individual)
print("Best metrics (accuracy, precision, recall, AUC, F1):", best_metrics)

# XGBoost model is trained with the best parameter configuration
best_num_boost_round = int(best_individual[0] * 1000)
best_max_depth = int(best_individual[1] * 10)
best_reg_alpha = np.clip(best_individual[2], 0, None)
best_min_child_weight = max(0, best_individual[3])
best_colsample_bynode = max(0, best_individual[4])
best_subsample = max(0, best_individual[5])
best_learning_rate = max(0, best_individual[6])
best_scale_pos_weight = max(0, best_individual[7])
params_best = {
    'objective': 'multi:softprob',
    'booster': 'gbtree',
    'num_class': 5,
    'max_depth': best_max_depth,
    'reg_alpha': best_reg_alpha,
    'min_child_weight': best_min_child_weight,
    'colsample_bynode': best_colsample_bynode,
    'subsample': best_subsample,
    'learning_rate': best_learning_rate
}
bst_best = xgb.train(params_best, dtrain, um_boost_round=best_num_boost_round, evals=[(dtest, 'eval'), (dtrain, 'train')],early_stopping_rounds=10)
y_pred_prob_best = bst_best.predict(dtest)
# Evaluate the model
y_pred_best= np.argmax(y_pred_prob_best, axis=1)
# Make sure that y_test is a label in integer format
y_test = y_test.astype(int)

acc_best = accuracy_score(y_test, y_pred_best)
pre_best = precision_score(y_test, y_pred_best, average='macro')
rec_best = recall_score(y_test, y_pred_best, average='macro')
auc_best = roc_auc_score(y_test, y_pred_prob_best, multi_class="ovr", average="macro")
f1_best = f1_score(y_test, y_pred_best, average='macro')
print("Best XGBoost Model Evaluation Metrics:")
print("Accuracy:", acc_best)
print("Precision:", pre_best)
print("Recall:", rec_best)
print("AUC:", auc_best)
print("F1 Score:", f1_best)

# Raster classification
TIF_PATH = r"C:\Users\liu\Downloads\tile_0_0.tif"
output_CLASS_TIF = r"C:\Users\liu\Downloads\output_class4.tif"

with rasterio.open(TIF_PATH) as src:
    profile = src.profile.copy()
    profile.update(dtype=rasterio.uint8, count=1, compress='lzw')

    # The total number of calculation windows is used for progress indications
    windows = list(src.block_windows(1))
    total_windows = len(windows)

    with rasterio.open(output_CLASS_TIF, 'w', **profile) as dst:
        for ji, window in tqdm(windows, total=total_windows, desc="Raster classification progress"):
            data = src.read(window=window)
            # data.shape = (bands, height, width)
            bands, height, width = data.shape
            data = data.reshape(bands, -1).transpose()  # shape: (num_pixels, bands)

            # Check that all band values are 0
            if np.all(data == 0, axis=1).all():
                # If all cells in the entire window are 0, set the output to 0
                out_image = np.zeros((height, width), dtype=np.uint8)
            else:
                # Handles cases where the cell value is 0
                data[data == 0] = np.nan  # Replace 0 with NaN as an invalid value

                # Create a DataFrame and use the feature column names at the time of training
                df = pd.DataFrame(data, columns=X.columns.tolist())
                dmatrix = xgb.DMatrix(df, feature_names=X.columns.tolist())

                # Prediction
                predictions = bst_best.predict(dmatrix)
                predictions = np.argmax(predictions, axis=1).astype(np.uint8)

                # Reshape to the original window shape
                out_image = predictions.reshape(height, width)

            # Write to the output raster
            dst.write(out_image, 1, window=window)

print("The raster classification is complete.")
