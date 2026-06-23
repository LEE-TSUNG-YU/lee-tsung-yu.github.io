# %% [markdown]
# Breast Cancer Wisconsin：五種分類模型比較

# %%
import logging
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna_integration import OptunaSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNClassifier
from xgboost import XGBClassifier


# %% [markdown]
# 1. 基本設定

# %%
RANDOM_STATE = 42
CV_FOLDS = 5
N_TRIALS = 30
TABPFN_N_TRIALS = 12

optuna.logging.set_verbosity(optuna.logging.WARNING)

# 若電腦有可供 PyTorch 使用的 NVIDIA GPU，可改成 "cuda"。
TABPFN_DEVICE = os.getenv("TABPFN_DEVICE", "cpu")


# %% [markdown]
# 2. 讀取資料

# %%
local_data_path = Path("data.csv")
kaggle_data_path = Path("/kaggle/input/datasets/mragpavank/breast-cancer/data.csv")

if local_data_path.exists():
    data_path = local_data_path
elif kaggle_data_path.exists():
    data_path = kaggle_data_path
else:
    raise FileNotFoundError("找不到 data.csv，請將檔案放在 code.py 相同資料夾。")

data = pd.read_csv(data_path)

print(f"資料來源：{data_path.resolve()}")
print(f"原始資料形狀：{data.shape}")
print(data.head())


# %% [markdown]
# 3. 建立特徵與目標

# %%
y = data["diagnosis"].map({"B": 0, "M": 1})

if y.isna().any():
    raise ValueError("diagnosis 欄位只能包含 B 與 M。")

X = data.drop(columns=["id", "diagnosis", "Unnamed: 32"], errors="ignore")

print(f"特徵資料形狀：{X.shape}")
print("\n目標類別筆數：")
print(y.value_counts().sort_index())


# %% [markdown]
# 4. 將資料切分為 train 70%、validation 15%、test 15%

# %%
# 先保留全部資料的 15% 作為 test。
X_train_validation, X_test, y_train_validation, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=RANDOM_STATE,
    stratify=y,
)

# 剩餘資料占 85%，因此用 0.15 / 0.85 切出原始資料的 15% 作為 validation。
validation_ratio = 0.15 / 0.85

X_train, X_validation, y_train, y_validation = train_test_split(
    X_train_validation,
    y_train_validation,
    test_size=validation_ratio,
    random_state=RANDOM_STATE,
    stratify=y_train_validation,
)

print(
    f"Train：{len(X_train)} 筆，"
    f"占全部資料 {len(X_train) / len(X):.2%}"
)
print(
    f"Validation：{len(X_validation)} 筆，"
    f"占全部資料 {len(X_validation) / len(X):.2%}"
)
print(
    f"Test：{len(X_test)} 筆，"
    f"占全部資料 {len(X_test) / len(X):.2%}"
)


# %% [markdown]
# 5. 建立交叉驗證策略

# %%
cv = StratifiedKFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

negative_count, positive_count = np.bincount(y_train)
scale_pos_weight = negative_count / positive_count

print(f"scale_pos_weight：{scale_pos_weight:.4f}")


# %% [markdown]
# 6. XGBoost ＋ OptunaSearchCV

# %%
xgb_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        (
            "model",
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                tree_method="hist",
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
        ),
    ]
)

xgb_param_distributions = {
    "model__n_estimators": IntDistribution(100, 500, step=50),
    "model__learning_rate": FloatDistribution(0.01, 0.2, log=True),
    "model__max_depth": IntDistribution(2, 8),
    "model__subsample": FloatDistribution(0.6, 1.0),
    "model__colsample_bytree": FloatDistribution(0.6, 1.0),
    "model__min_child_weight": IntDistribution(1, 10),
}

xgb_optuna_search = OptunaSearchCV(
    estimator=xgb_pipeline,
    param_distributions=xgb_param_distributions,
    scoring="roc_auc",
    cv=cv,
    n_jobs=1,
    n_trials=N_TRIALS,
    random_state=RANDOM_STATE,
    refit=True,
    verbose=1,
    error_score="raise",
)

xgb_optuna_search.fit(X_train, y_train)

print("XGBoost 最佳參數：", xgb_optuna_search.best_params_)
print(f"XGBoost 最佳 CV ROC-AUC：{xgb_optuna_search.best_score_:.4f}")


# %% [markdown]
# 7. CatBoost ＋ OptunaSearchCV

# %%
catboost_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        (
            "model",
            CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="AUC",
                verbose=0,
                random_seed=RANDOM_STATE,
                thread_count=1,
            ),
        ),
    ]
)

catboost_param_distributions = {
    "model__iterations": IntDistribution(100, 600, step=50),
    "model__learning_rate": FloatDistribution(0.01, 0.2, log=True),
    "model__depth": IntDistribution(3, 8),
    "model__l2_leaf_reg": FloatDistribution(1.0, 10.0, log=True),
    "model__random_strength": FloatDistribution(0.0, 2.0),
}

catboost_optuna_search = OptunaSearchCV(
    estimator=catboost_pipeline,
    param_distributions=catboost_param_distributions,
    scoring="roc_auc",
    cv=cv,
    n_jobs=1,
    n_trials=N_TRIALS,
    random_state=RANDOM_STATE,
    refit=True,
    verbose=1,
    error_score="raise",
)

catboost_optuna_search.fit(X_train, y_train)

print("CatBoost 最佳參數：", catboost_optuna_search.best_params_)
print(f"CatBoost 最佳 CV ROC-AUC：{catboost_optuna_search.best_score_:.4f}")


# %% [markdown]
# 8. LightGBM ＋ OptunaSearchCV

# %%
lightgbm_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        (
            "model",
            LGBMClassifier(
                objective="binary",
                class_weight="balanced",
                random_state=RANDOM_STATE,
                verbosity=-1,
                n_jobs=1,
            ),
        ),
    ]
)

lightgbm_param_distributions = {
    "model__n_estimators": IntDistribution(100, 500, step=50),
    "model__learning_rate": FloatDistribution(0.01, 0.2, log=True),
    "model__num_leaves": IntDistribution(15, 63),
    "model__max_depth": IntDistribution(3, 10),
    "model__min_child_samples": IntDistribution(5, 40),
    "model__subsample": FloatDistribution(0.6, 1.0),
    "model__colsample_bytree": FloatDistribution(0.6, 1.0),
}

lightgbm_optuna_search = OptunaSearchCV(
    estimator=lightgbm_pipeline,
    param_distributions=lightgbm_param_distributions,
    scoring="roc_auc",
    cv=cv,
    n_jobs=1,
    n_trials=N_TRIALS,
    random_state=RANDOM_STATE,
    refit=True,
    verbose=1,
    error_score="raise",
)

lightgbm_optuna_search.fit(X_train, y_train)

print("LightGBM 最佳參數：", lightgbm_optuna_search.best_params_)
print(f"LightGBM 最佳 CV ROC-AUC：{lightgbm_optuna_search.best_score_:.4f}")


# %% [markdown]
# 9. MLP ＋ OptunaSearchCV

# %%
# MLP 對特徵尺度敏感，因此在 Pipeline 中加入 StandardScaler。
mlp_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "model",
            MLPClassifier(
                max_iter=2000,
                early_stopping=True,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

mlp_param_distributions = {
    "model__hidden_layer_sizes": CategoricalDistribution(
        [(50,), (100,), (100, 50), (128, 64)]
    ),
    "model__activation": CategoricalDistribution(["relu", "tanh"]),
    "model__alpha": FloatDistribution(1e-5, 1e-2, log=True),
    "model__learning_rate_init": FloatDistribution(1e-4, 1e-2, log=True),
    "model__batch_size": CategoricalDistribution([16, 32, 64]),
}

mlp_optuna_search = OptunaSearchCV(
    estimator=mlp_pipeline,
    param_distributions=mlp_param_distributions,
    scoring="roc_auc",
    cv=cv,
    n_jobs=1,
    n_trials=N_TRIALS,
    random_state=RANDOM_STATE,
    refit=True,
    verbose=1,
    error_score="raise",
)

mlp_optuna_search.fit(X_train, y_train)

print("MLP 最佳參數：", mlp_optuna_search.best_params_)
print(f"MLP 最佳 CV ROC-AUC：{mlp_optuna_search.best_score_:.4f}")


# %% [markdown]
# 10. TabPFN ＋ OptunaSearchCV

# %%
# 第一次執行 TabPFN 時，套件可能需要從 Hugging Face 下載預訓練模型。
tabpfn_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        (
            "model",
            TabPFNClassifier(
                device=TABPFN_DEVICE,
                random_state=RANDOM_STATE,
                show_progress_bar=False,
            ),
        ),
    ]
)

tabpfn_param_distributions = {
    "model__n_estimators": IntDistribution(4, 12, step=2),
    "model__softmax_temperature": FloatDistribution(0.7, 1.2),
    "model__balance_probabilities": CategoricalDistribution([False, True]),
}

tabpfn_optuna_search = OptunaSearchCV(
    estimator=tabpfn_pipeline,
    param_distributions=tabpfn_param_distributions,
    scoring="roc_auc",
    cv=cv,
    n_jobs=1,
    n_trials=TABPFN_N_TRIALS,
    random_state=RANDOM_STATE,
    refit=True,
    verbose=1,
    error_score="raise",
)

tabpfn_optuna_search.fit(X_train, y_train)

print("TabPFN 最佳參數：", tabpfn_optuna_search.best_params_)
print(f"TabPFN 最佳 CV ROC-AUC：{tabpfn_optuna_search.best_score_:.4f}")


# %% [markdown]
# 11. 使用 validation 與 test 比較五種模型

# %%
optuna_search_models = {
    "XGBoost": xgb_optuna_search,
    "CatBoost": catboost_optuna_search,
    "LightGBM": lightgbm_optuna_search,
    "MLP": mlp_optuna_search,
    "TabPFN": tabpfn_optuna_search,
}

results = []

for model_name, optuna_search in optuna_search_models.items():
    validation_prediction = optuna_search.predict(X_validation)
    validation_probability = optuna_search.predict_proba(X_validation)[:, 1]

    test_prediction = optuna_search.predict(X_test)
    test_probability = optuna_search.predict_proba(X_test)[:, 1]

    model_result = {
        "model": model_name,
        "best_cv_roc_auc": optuna_search.best_score_,
        "validation_accuracy": accuracy_score(
            y_validation, validation_prediction
        ),
        "validation_precision": precision_score(
            y_validation, validation_prediction, zero_division=0
        ),
        "validation_recall": recall_score(
            y_validation, validation_prediction, zero_division=0
        ),
        "validation_f1": f1_score(
            y_validation, validation_prediction, zero_division=0
        ),
        "validation_roc_auc": roc_auc_score(
            y_validation, validation_probability
        ),
        "test_accuracy": accuracy_score(y_test, test_prediction),
        "test_precision": precision_score(
            y_test, test_prediction, zero_division=0
        ),
        "test_recall": recall_score(
            y_test, test_prediction, zero_division=0
        ),
        "test_f1": f1_score(y_test, test_prediction, zero_division=0),
        "test_roc_auc": roc_auc_score(y_test, test_probability),
        "best_params": optuna_search.best_params_,
    }

    results.append(model_result)

    print(f"\n{'=' * 70}")
    print(f"{model_name} Test classification report")
    print(
        classification_report(
            y_test,
            test_prediction,
            target_names=["Benign", "Malignant"],
            digits=4,
            zero_division=0,
        )
    )


# %% [markdown]
# 12. 顯示並儲存模型比較結果

# %%
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="validation_roc_auc",
    ascending=False,
).reset_index(drop=True)

display_columns = [
    "model",
    "best_cv_roc_auc",
    "validation_accuracy",
    "validation_roc_auc",
    "test_accuracy",
    "test_f1",
    "test_roc_auc",
]

print("\n模型比較結果：")
print(
    results_df[display_columns].to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

output_path = Path("model_results.csv")
results_df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\n完整結果已儲存至：{output_path.resolve()}")
