import rasterio
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from tqdm import tqdm  # 用于进度指示

# 读取样本点 CSV 文件（包含特征和标签）
samples = pd.read_csv(r"C:\Users\liu\Downloads\new_labeled_data.csv")

# 提取特征和标签
X = samples.drop(['landcover'], axis=1)  # 假设 'landcover' 列是标签
y = samples['landcover']

# 数据集分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 训练 XGBoost 模型
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X_train.columns.tolist())
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=X_train.columns.tolist())

params = {
    'objective': 'multi:softprob',
    'booster': 'gbtree',
    'num_class': 5,  # 根据分类数量调整
    'n_estimators':200,
    'max_depth': 3,
    'reg_alpha': 0.2,
    'min_child_weight': 0.3,
    'colsample_bynode': 0.3,
    'subsample': 0.2,
    'learning_rate': 0.01,
    'scale_pos_weight': 0.3
}

# 训练模型
bst = xgb.train(params, dtrain, evals=[(dtest, 'eval'), (dtrain, 'train')], early_stopping_rounds=10)
y_pred_prob = bst.predict(dtest)
# 评估模型
y_pred = np.argmax(y_pred_prob, axis=1)

# 确保 y_test 是整数格式的标签
y_test = y_test.astype(int)
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'AUC: {auc}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(report)

# 栅格分类
TIF_PATH = r"C:\Users\liu\Downloads\tile_0_0.tif"
output_CLASS_TIF = r"C:\Users\liu\Downloads\output_class2.tif"

with rasterio.open(TIF_PATH) as src:
    profile = src.profile.copy()
    profile.update(dtype=rasterio.uint8, count=1, compress='lzw')

    # 计算窗口总数用于进度指示
    windows = list(src.block_windows(1))
    total_windows = len(windows)

    with rasterio.open(output_CLASS_TIF, 'w', **profile) as dst:
        for ji, window in tqdm(windows, total=total_windows, desc="栅格分类进度"):
            data = src.read(window=window)
            # data.shape = (bands, height, width)
            bands, height, width = data.shape
            data = data.reshape(bands, -1).transpose()  # shape: (num_pixels, bands)

            # 检查是否所有波段值都为 0
            if np.all(data == 0, axis=1).all():
                # 如果整个窗口所有像元都是 0，设置输出为 0
                out_image = np.zeros((height, width), dtype=np.uint8)
            else:
                # 处理像元值为0的情况
                data[data == 0] = np.nan  # 将 0 替换为 NaN，作为无效值

                # 创建 DataFrame，使用训练时的特征列名
                df = pd.DataFrame(data, columns=X.columns.tolist())
                dmatrix = xgb.DMatrix(df, feature_names=X.columns.tolist())

                # 预测
                predictions = bst.predict(dmatrix)
                predictions = np.argmax(predictions, axis=1).astype(np.uint8)  # 多分类预测

                # 重塑为原窗口形状
                out_image = predictions.reshape(height, width)

            # 写入输出栅格
            dst.write(out_image, 1, window=window)

print("栅格分类完成。")
