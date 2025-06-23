import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

class StackingEnsemble:
    def __init__(self, base_models, meta_model, cv_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.cv_folds = cv_folds
        self.fitted_base_models = {}
        
    def fit(self, X, y):
        # Подготовка мета-признаков
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        for i, (name, model) in enumerate(self.base_models.items()):
            print(f"Обучение {name}...")
            
            # Out-of-fold предсказания для мета-признаков
            oof_predictions = np.zeros(X.shape[0])
            
            fold_models = []
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
                y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Клонируем модель для каждого фолда
                fold_model = self._clone_model(model)
                fold_model.fit(X_fold_train, y_fold_train)
                
                # Предсказания для валидационной части
                val_pred = fold_model.predict_proba(X_fold_val)[:, 1]
                oof_predictions[val_idx] = val_pred
                
                fold_models.append(fold_model)
            
            meta_features[:, i] = oof_predictions
            self.fitted_base_models[name] = fold_models
            
            # Качество базовой модели
            oof_score = log_loss(y, oof_predictions)
            print(f"{name} OOF Log Loss: {oof_score:.4f}")
        
        # Обучение мета-модели
        print("Обучение мета-модели...")
        self.meta_model.fit(meta_features, y)
        
        return self
    
    def predict_proba(self, X_test):
        # Предсказания базовых моделей
        meta_features_test = np.zeros((X_test.shape[0], len(self.base_models)))
        
        for i, (name, fold_models) in enumerate(self.fitted_base_models.items()):
            # Усредняем предсказания всех фолдов
            fold_predictions = []
            for model in fold_models:
                pred = model.predict_proba(X_test)[:, 1]
                fold_predictions.append(pred)
            
            meta_features_test[:, i] = np.mean(fold_predictions, axis=0)
        
        # Предсказание мета-модели
        return self.meta_model.predict_proba(meta_features_test)
    
    def predict(self, X_test):
        return np.argmax(self.predict_proba(X_test), axis=1)
    
    def _clone_model(self, model):
        # Простое клонирование модели с теми же параметрами
        model_class = type(model)
        params = model.get_params()
        return model_class(**params)

# Многоуровневый стекинг
class MultiLevelStacking:
    def __init__(self):
        # Уровень 1: разнообразные базовые модели
        self.level1_models = {
            'xgb': xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
            'lgb': lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1),
            'cat': CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1, random_seed=42, verbose=False),
            'rf': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        }
        
        # Уровень 2: бустинг с разными параметрами
        self.level2_models = {
            'xgb_deep': xgb.XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.05, random_state=42),
            'lgb_deep': lgb.LGBMClassifier(n_estimators=200, max_depth=8, learning_rate=0.05, random_state=42, verbose=-1)
        }
        
        # Финальная мета-модель
        self.meta_model = LogisticRegression(random_state=42)
        
    def fit(self, X, y):
        # Обучение первого уровня
        print("=== Уровень 1 ===")
        self.stacking_l1 = StackingEnsemble(self.level1_models, LogisticRegression(random_state=42))
        self.stacking_l1.fit(X, y)
        
        # Получение мета-признаков первого уровня
        meta_features_l1 = np.zeros((X.shape[0], len(self.level1_models)))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for i, (name, fold_models) in enumerate(self.stacking_l1.fitted_base_models.items()):
            oof_predictions = np.zeros(X.shape[0])
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                val_pred = fold_models[fold].predict_proba(X.iloc[val_idx])[:, 1]
                oof_predictions[val_idx] = val_pred
            meta_features_l1[:, i] = oof_predictions
        
        # Обучение второго уровня
        print("\n=== Уровень 2 ===")
        # Объединяем исходные признаки с мета-признаками первого уровня
        X_level2 = np.hstack([X.values, meta_features_l1])
        X_level2 = pd.DataFrame(X_level2, columns=list(X.columns) + [f'meta_{i}' for i in range(meta_features_l1.shape[1])])
        
        self.stacking_l2 = StackingEnsemble(self.level2_models, self.meta_model)
        self.stacking_l2.fit(X_level2, y)
        
        return self
    
    def predict_proba(self, X_test):
        # Предсказания первого уровня
        meta_features_l1_test = self.stacking_l1.predict_proba(X_test)
        
        # Объединяем с исходными признаками для второго уровня
        X_level2_test = np.hstack([X_test.values, meta_features_l1_test])
        X_level2_test = pd.DataFrame(X_level2_test, columns=list(X_test.columns) + [f'meta_{i}' for i in range(meta_features_l1_test.shape[1])])
        
        # Предсказания второго уровня
        return self.stacking_l2.predict_proba(X_level2_test)
    
    def predict(self, X_test):
        return np.argmax(self.predict_proba(X_test), axis=1)

# Загрузка данных
def load_classification_data():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=2000, n_features=20, n_informative=15, 
                             n_redundant=3, n_classes=2, random_state=42)
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    df['target'] = y
    
    X_test, _ = make_classification(n_samples=500, n_features=20, n_informative=15,
                                 n_redundant=3, n_classes=2, random_state=123)
    test_df = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(20)])
    
    return df, test_df

# Сравнение разных подходов
def compare_approaches(X, y, X_test):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = {}
    
    # 1. Простой XGBoost
    print("=== Простой XGBoost ===")
    simple_xgb = xgb.XGBClassifier(n_estimators=100, random_state=42)
    simple_xgb.fit(X_train, y_train)
    val_pred = simple_xgb.predict(X_val)
    results['Simple XGBoost'] = accuracy_score(y_val, val_pred)
    
    # 2. Простой стекинг
    print("\n=== Простой стекинг ===")
    base_models = {
        'xgb': xgb.XGBClassifier(n_estimators=100, random_state=42),
        'lgb': lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
        'cat': CatBoostClassifier(iterations=100, random_seed=42, verbose=False)
    }
    
    simple_stacking = StackingEnsemble(base_models, LogisticRegression(random_state=42))
    simple_stacking.fit(X_train, y_train)
    val_pred = simple_stacking.predict(X_val)
    results['Simple Stacking'] = accuracy_score(y_val, val_pred)
    
    # 3. Многоуровневый стекинг
    print("\n=== Многоуровневый стекинг ===")
    multilevel = MultiLevelStacking()
    multilevel.fit(X_train, y_train)
    val_pred = multilevel.predict(X_val)
    results['Multi-level Stacking'] = accuracy_score(y_val, val_pred)
    
    print("\n=== Результаты сравнения ===")
    for name, score in results.items():
        print(f"{name}: {score:.4f}")
    
    # Финальные предсказания лучшей модели
    best_model_name = max(results, key=results.get)
    if best_model_name == 'Multi-level Stacking':
        final_pred = multilevel.predict_proba(X_test)[:, 1]
    elif best_model_name == 'Simple Stacking':
        final_pred = simple_stacking.predict_proba(X_test)[:, 1]
    else:
        final_pred = simple_xgb.predict_proba(X_test)[:, 1]
    
    return final_pred

# Основная функция
def main():
    print("Загрузка данных...")
    df, test_df = load_classification_data()
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    print("Сравнение подходов...")
    final_predictions = compare_approaches(X, y, test_df)
    
    # Сохранение результатов
    submission = pd.DataFrame({
        'id': range(len(final_predictions)),
        'prediction': final_predictions
    })
    submission.to_csv('stacking_submission.csv', index=False)
    print("\nРезультаты сохранены в stacking_submission.csv")

if __name__ == "__main__":
    main()