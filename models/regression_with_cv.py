import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
def load_data():
    # Создаем синтетические данные для регрессии
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=1000, n_features=15, noise=0.1, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(15)])
    df['target'] = y
    
    X_test, _ = make_regression(n_samples=200, n_features=15, noise=0.1, random_state=123)
    test_df = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(15)])
    
    return df, test_df

# Предобработка для регрессии
def preprocess_regression(df, test_df, target_col='target'):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Масштабирование только для некоторых моделей
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    test_scaled = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns)
    
    return X, y, test_df, X_scaled, test_scaled

# Кросс-валидация с разными метриками
def cross_validate_models(X, y, cv_folds=5):
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    models = {
        'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1),
        'CatBoost': CatBoostRegressor(iterations=100, depth=6, learning_rate=0.1, random_seed=42, verbose=False)
    }
    
    results = {}
    
    for name, model in models.items():
        # RMSE
        rmse_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error')
        # MAE
        mae_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_error')
        
        results[name] = {
            'RMSE': rmse_scores,
            'MAE': mae_scores,
            'RMSE_mean': rmse_scores.mean(),
            'MAE_mean': mae_scores.mean(),
            'model': model
        }
        
        print(f"{name}:")
        print(f"  RMSE: {rmse_scores.mean():.4f} (+/- {rmse_scores.std() * 2:.4f})")
        print(f"  MAE:  {mae_scores.mean():.4f} (+/- {mae_scores.std() * 2:.4f})")
    
    return results

# Оптимизация гиперпараметров
def optimize_hyperparameters(X, y):
    from sklearn.model_selection import RandomizedSearchCV
    
    # Параметры для XGBoost
    xgb_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    xgb_model = xgb.XGBRegressor(random_state=42)
    xgb_search = RandomizedSearchCV(
        xgb_model, xgb_params, n_iter=10, cv=3, 
        scoring='neg_root_mean_squared_error', random_state=42
    )
    xgb_search.fit(X, y)
    
    print(f"Лучшие параметры XGBoost: {xgb_search.best_params_}")
    print(f"Лучший RMSE: {-xgb_search.best_score_:.4f}")
    
    return xgb_search.best_estimator_

# Обучение финальных моделей
def train_final_models(X, y):
    models = {}
    
    # XGBoost с оптимизированными параметрами
    best_xgb = optimize_hyperparameters(X, y)
    models['xgb_optimized'] = best_xgb
    
    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=150,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X, y)
    models['lgb'] = lgb_model
    
    # CatBoost
    cat_model = CatBoostRegressor(
        iterations=150,
        depth=7,
        learning_rate=0.05,
        random_seed=42,
        verbose=False
    )
    cat_model.fit(X, y)
    models['catboost'] = cat_model
    
    return models

# Ансамбль с весами
def weighted_ensemble(models, X_test, weights=None):
    if weights is None:
        weights = [1/len(models)] * len(models)
    
    predictions = []
    for name, model in models.items():
        pred = model.predict(X_test)
        predictions.append(pred)
    
    # Взвешенное усреднение
    ensemble_pred = np.average(predictions, axis=0, weights=weights)
    
    return ensemble_pred

# Визуализация важности признаков
def plot_feature_importance(models, feature_names):
    fig, axes = plt.subplots(1, len(models), figsize=(15, 5))
    
    for i, (name, model) in enumerate(models.items()):
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            indices = np.argsort(importance)[::-1][:10]
            
            axes[i].bar(range(10), importance[indices])
            axes[i].set_title(f'{name} - Важность признаков')
            axes[i].set_xticks(range(10))
            axes[i].set_xticklabels([feature_names[j] for j in indices], rotation=45)
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()

# Основная функция
def main():
    print("Загрузка данных...")
    df, test_df = load_data()
    
    print("Предобработка...")
    X, y, X_test, X_scaled, test_scaled = preprocess_regression(df, test_df)
    
    print("Кросс-валидация...")
    cv_results = cross_validate_models(X, y)
    
    print("\nОбучение финальных моделей...")
    final_models = train_final_models(X, y)
    
    print("Создание ансамбля...")
    # Веса основаны на производительности CV
    weights = [0.4, 0.3, 0.3]  # Можно настроить на основе результатов CV
    ensemble_pred = weighted_ensemble(final_models, X_test, weights)
    
    # Визуализация
    plot_feature_importance(final_models, X.columns)
    
    # Сохранение результатов
    submission = pd.DataFrame({
        'id': range(len(ensemble_pred)),
        'target': ensemble_pred
    })
    submission.to_csv('regression_submission.csv', index=False)
    print("Результаты сохранены в regression_submission.csv")

if __name__ == "__main__":
    main()