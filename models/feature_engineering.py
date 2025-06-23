import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineering:
    def __init__(self):
        self.polynomial_features = None
        self.scaler = StandardScaler()
        self.selector = None
        self.pca = None
        self.feature_names = []
        
    def create_polynomial_features(self, X, degree=2):
        """Создание полиномиальных признаков"""
        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
        X_poly = poly.fit_transform(X)
        
        # Получаем названия признаков
        feature_names = poly.get_feature_names_out(X.columns)
        
        self.polynomial_features = poly
        return pd.DataFrame(X_poly, columns=feature_names, index=X.index)
    
    def create_interaction_features(self, X):
        """Создание взаимодействий между признаками"""
        interactions = pd.DataFrame(index=X.index)
        
        # Попарные произведения
        for i in range(len(X.columns)):
            for j in range(i+1, len(X.columns)):
                col1, col2 = X.columns[i], X.columns[j]
                interactions[f'{col1}_x_{col2}'] = X[col1] * X[col2]
        
        # Соотношения
        for i in range(len(X.columns)):
            for j in range(i+1, len(X.columns)):
                col1, col2 = X.columns[i], X.columns[j]
                # Избегаем деления на ноль
                mask = X[col2] != 0
                interactions[f'{col1}_div_{col2}'] = 0
                interactions.loc[mask, f'{col1}_div_{col2}'] = X.loc[mask, col1] / X.loc[mask, col2]
        
        return interactions
    
    def create_statistical_features(self, X):
        """Создание статистических признаков"""
        stat_features = pd.DataFrame(index=X.index)
        
        # Статистики по строкам
        stat_features['row_mean'] = X.mean(axis=1)
        stat_features['row_std'] = X.std(axis=1)
        stat_features['row_min'] = X.min(axis=1)
        stat_features['row_max'] = X.max(axis=1)
        stat_features['row_median'] = X.median(axis=1)
        stat_features['row_skew'] = X.skew(axis=1)
        stat_features['row_kurtosis'] = X.kurtosis(axis=1)
        
        # Количество нулей и отрицательных значений
        stat_features['zero_count'] = (X == 0).sum(axis=1)
        stat_features['negative_count'] = (X < 0).sum(axis=1)
        
        return stat_features
    
    def create_clustering_features(self, X, n_clusters=5):
        """Создание признаков на основе кластеризации"""
        from sklearn.cluster import KMeans
        
        cluster_features = pd.DataFrame(index=X.index)
        
        # K-means кластеризация
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        cluster_features['kmeans_cluster'] = clusters
        
        # Расстояния до центроидов
        distances = kmeans.transform(X)
        for i in range(n_clusters):
            cluster_features[f'distance_to_cluster_{i}'] = distances[:, i]
        
        return cluster_features, kmeans
    
    def select_features(self, X, y, method='mutual_info', k=50):
        """Отбор признаков"""
        if method == 'mutual_info':
            from sklearn.feature_selection import mutual_info_classif
            selector = SelectKBest(mutual_info_classif, k=k)
        elif method == 'f_classif':
            selector = SelectKBest(f_classif, k=k)
        elif method == 'rfe':
            estimator = xgb.XGBClassifier(random_state=42)
            selector = RFE(estimator, n_features_to_select=k)
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
        
        self.selector = selector
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def create_target_encoding(self, X, y, categorical_cols):
        """Target encoding для категориальных признаков"""
        from sklearn.model_selection import KFold
        
        encoded_features = X.copy()
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for col in categorical_cols:
            if col in X.columns:
                encoded_col = np.zeros(len(X))
                
                for train_idx, val_idx in kf.split(X):
                    # Вычисляем среднее по целевой переменной для каждой категории
                    target_mean = y.iloc[train_idx].groupby(X[col].iloc[train_idx]).mean()
                    
                    # Применяем к валидационной выборке
                    encoded_col[val_idx] = X[col].iloc[val_idx].map(target_mean).fillna(y.mean())
                
                encoded_features[f'{col}_target_encoded'] = encoded_col
        
        return encoded_features

# Комплексная обработка данных
def comprehensive_feature_engineering(df, test_df, target_col='target'):
    """Полный пайплайн обработки признаков"""
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    fe = AdvancedFeatureEngineering()
    
    print("Исходные признаки:", X.shape[1])
    
    # 1. Создание полиномиальных признаков (только для небольшого количества)
    if X.shape[1] <= 10:
        X_poly = fe.create_polynomial_features(X, degree=2)
        test_poly = fe.polynomial_features.transform(test_df)
        test_poly = pd.DataFrame(test_poly, columns=X_poly.columns, index=test_df.index)
        print("После полиномиальных признаков:", X_poly.shape[1])
    else:
        X_poly = X.copy()
        test_poly = test_df.copy()
    
    # 2. Статистические признаки
    X_stat = fe.create_statistical_features(X)
    test_stat = fe.create_statistical_features(test_df)
    print("Статистических признаков:", X_stat.shape[1])
    
    # 3. Кластерные признаки
    X_cluster, kmeans_model = fe.create_clustering_features(X)
    test_cluster = pd.DataFrame(index=test_df.index)
    test_clusters = kmeans_model.predict(test_df)
    test_cluster['kmeans_cluster'] = test_clusters
    test_distances = kmeans_model.transform(test_df)
    for i in range(test_distances.shape[1]):
        test_cluster[f'distance_to_cluster_{i}'] = test_distances[:, i]
    
    print("Кластерных признаков:", X_cluster.shape[1])
    
    # 4. Объединение всех признаков
    X_combined = pd.concat([X_poly, X_stat, X_cluster], axis=1)
    test_combined = pd.concat([test_poly, test_stat, test_cluster], axis=1)
    
    print("Всего признаков после объединения:", X_combined.shape[1])
    
    # 5. Отбор признаков
    X_selected = fe.select_features(X_combined, y, method='f_classif', k=min(100, X_combined.shape[1]))
    test_selected = fe.selector.transform(test_combined)
    test_selected = pd.DataFrame(test_selected, columns=X_selected.columns, index=test_df.index)
    
    print("После отбора признаков:", X_selected.shape[1])
    
    return X_selected, test_selected, y

# Обучение с engineered признаками
def train_with_engineered_features(X, y, X_test):
    """Обучение моделей с обработанными признаками"""
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {}
    
    # XGBoost с настройками для большого количества признаков
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    models['xgb'] = xgb_model
    
    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    models['lgb'] = lgb_model
    
    # CatBoost
    cat_model = CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=False
    )
    cat_model.fit(X_train, y_train)
    models['catboost'] = cat_model
    
    # Валидация
    for name, model in models.items():
        val_pred = model.predict(X_val)
        val_proba = model.predict_proba(X_val)[:, 1]
        
        from sklearn.metrics import accuracy_score, roc_auc_score
        acc = accuracy_score(y_val, val_pred)
        auc = roc_auc_score(y_val, val_proba)
        
        print(f'{name} - Accuracy: {acc:.4f}, AUC: {auc:.4f}')
    
    # Ансамбль
    test_predictions = []
    for model in models.values():
        pred = model.predict_proba(X_test)[:, 1]
        test_predictions.append(pred)
    
    ensemble_pred = np.mean(test_predictions, axis=0)
    
    return ensemble_pred, models

# Анализ важности признаков
def analyze_feature_importance(models, feature_names, top_k=20):
    """Анализ важности признаков"""
    
    importance_df = pd.DataFrame()
    
    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            temp_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance,
                'model': name
            })
            importance_df = pd.concat([importance_df, temp_df])
    
    # Средняя важность по всем моделям
    avg_importance = importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
    
    print(f"\nТоп {top_k} самых важных признаков:")
    for i, (feature, importance) in enumerate(avg_importance.head(top_k).items(), 1):
        print(f"{i:2d}. {feature}: {importance:.4f}")
    
    return avg_importance

# Основная функция
def main():
    # Создание синтетических данных с разными типами признаков
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=1500, 
        n_features=15, 
        n_informative=10,
        n_redundant=3,
        n_classes=2, 
        random_state=42
    )
    
    # Добавляем шум и категориальные признаки
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(15)])
    df['cat_feature_1'] = np.random.choice(['A', 'B', 'C', 'D'], size=len(df))
    df['cat_feature_2'] = np.random.choice(['X', 'Y', 'Z'], size=len(df))
    df['target'] = y
    
    # Тестовые данные
    X_test, _ = make_classification(
        n_samples=300, 
        n_features=15, 
        n_informative=10,
        n_redundant=3,
        n_classes=2, 
        random_state=123
    )
    test_df = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(15)])
    test_df['cat_feature_1'] = np.random.choice(['A', 'B', 'C', 'D'], size=len(test_df))
    test_df['cat_feature_2'] = np.random.choice(['X', 'Y', 'Z'], size=len(test_df))
    
    print("Комплексная обработка признаков...")
    X_engineered, test_engineered, y = comprehensive_feature_engineering(df, test_df)
    
    print("\nОбучение моделей...")
    final_predictions, trained_models = train_with_engineered_features(X_engineered, y, test_engineered)
    
    print("\nАнализ важности признаков...")
    analyze_feature_importance(trained_models, X_engineered.columns)
    
    # Сохранение результатов
    submission = pd.DataFrame({
        'id': range(len(final_predictions)),
        'prediction': final_predictions
    })
    submission.to_csv('engineered_features_submission.csv', index=False)
    print("\nРезультаты сохранены в engineered_features_submission.csv")

if __name__ == "__main__":
    main()