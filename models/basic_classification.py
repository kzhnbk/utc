import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
def load_data():
    # Пример загрузки - замените на ваш датасет
    # df = pd.read_csv('train.csv')
    # test_df = pd.read_csv('test.csv')
    
    # Создаем синтетические данные для примера
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                             n_redundant=5, n_classes=3, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    df['target'] = y
    
    # Тестовые данные
    X_test, _ = make_classification(n_samples=200, n_features=20, n_informative=15,
                                 n_redundant=5, n_classes=3, random_state=123)
    test_df = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(20)])
    
    return df, test_df

# Предобработка
def preprocess_data(df, test_df, target_col='target'):
    # Разделение признаков и целевой переменной
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Кодирование категориальных признаков
    cat_features = X.select_dtypes(include=['object']).columns
    for col in cat_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        if col in test_df.columns:
            test_df[col] = le.transform(test_df[col].astype(str))
    
    # Обработка пропусков
    X = X.fillna(X.median())
    test_df = test_df.fillna(test_df.median())
    
    return X, y, test_df, cat_features

# Обучение моделей
def train_models(X, y, cat_features):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {}
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss'
    )
    xgb_model.fit(X_train, y_train)
    models['xgb'] = xgb_model
    
    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    models['lgb'] = lgb_model
    
    # CatBoost
    cat_model = CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        random_seed=42,
        verbose=False
    )
    cat_model.fit(X_train, y_train, cat_features=list(cat_features))
    models['catboost'] = cat_model
    
    # Валидация
    for name, model in models.items():
        val_pred = model.predict(X_val)
        acc = accuracy_score(y_val, val_pred)
        print(f'{name} Validation Accuracy: {acc:.4f}')
    
    return models

# Создание ансамбля
def create_ensemble(models, X_test):
    predictions = []
    
    for name, model in models.items():
        pred = model.predict_proba(X_test)
        predictions.append(pred)
    
    # Простое усреднение
    ensemble_pred = np.mean(predictions, axis=0)
    final_pred = np.argmax(ensemble_pred, axis=1)
    
    return final_pred, ensemble_pred

# Основная функция
def main():
    print("Загрузка данных...")
    df, test_df = load_data()
    
    print("Предобработка...")
    X, y, X_test, cat_features = preprocess_data(df, test_df)
    
    print("Обучение моделей...")
    models = train_models(X, y, cat_features)
    
    print("Создание предсказаний...")
    final_pred, proba_pred = create_ensemble(models, X_test)
    
    # Сохранение результатов
    submission = pd.DataFrame({
        'id': range(len(final_pred)),
        'prediction': final_pred
    })
    submission.to_csv('submission.csv', index=False)
    print("Результаты сохранены в submission.csv")

if __name__ == "__main__":
    main()