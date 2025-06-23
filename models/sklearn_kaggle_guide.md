# Sklearn для Kaggle: Ключевые модули и практические примеры

## 1. sklearn.impute - Заполнение пропущенных значений

### SimpleImputer - базовые стратегии
```python
from sklearn.impute import SimpleImputer
import numpy as np

# Числовые данные
numeric_imputer = SimpleImputer(strategy='median')  # mean, median, most_frequent, constant
X_numeric_filled = numeric_imputer.fit_transform(X_numeric)

# Категориальные данные
categorical_imputer = SimpleImputer(strategy='most_frequent')
X_cat_filled = categorical_imputer.fit_transform(X_categorical)

# Константное заполнение
constant_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
```

### KNNImputer - интеллектуальное заполнение
```python
from sklearn.impute import KNNImputer

# Заполнение на основе K ближайших соседей
knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
X_filled = knn_imputer.fit_transform(X)

# Для разных типов данных
knn_imputer_uniform = KNNImputer(n_neighbors=3, weights='uniform')
```

### IterativeImputer - многократное заполнение
```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# Использование модели для предсказания пропусков
iterative_imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=10, random_state=42),
    max_iter=10,
    random_state=42
)
X_filled = iterative_imputer.fit_transform(X)
```

## 2. sklearn.preprocessing - Предобработка данных

### Масштабирование признаков
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer

# Стандартизация (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Нормализация в диапазон [0,1]
minmax_scaler = MinMaxScaler()
X_normalized = minmax_scaler.fit_transform(X)

# Робастное масштабирование (устойчиво к выбросам)
robust_scaler = RobustScaler()
X_robust = robust_scaler.fit_transform(X)

# Преобразование к нормальному распределению
power_transformer = PowerTransformer(method='yeo-johnson')
X_transformed = power_transformer.fit_transform(X)
```

### Кодирование категориальных признаков
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

# Label Encoding для target переменной
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# One-Hot Encoding
onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_onehot = onehot_encoder.fit_transform(X_categorical)

# Ordinal Encoding для порядковых признаков
ordinal_encoder = OrdinalEncoder(categories=[['low', 'medium', 'high']])
X_ordinal = ordinal_encoder.fit_transform(X_ordinal_features)
```

### Создание полиномиальных признаков
```python
from sklearn.preprocessing import PolynomialFeatures

# Создание полиномиальных признаков
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_poly = poly.fit_transform(X)

# Только взаимодействия без степеней
poly_interactions = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
```

## 3. sklearn.feature_selection - Отбор признаков

### Статистические методы
```python
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, chi2

# Для классификации
selector_classif = SelectKBest(score_func=f_classif, k=10)
X_selected = selector_classif.fit_transform(X, y)

# Для регрессии
selector_regr = SelectKBest(score_func=f_regression, k='all')
scores = selector_regr.fit(X, y).scores_

# Chi-квадрат для категориальных признаков
chi2_selector = SelectKBest(score_func=chi2, k=15)
```

### Рекурсивное исключение признаков (RFE)
```python
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier

# RFE с фиксированным количеством признаков
estimator = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator, n_features_to_select=10, step=1)
X_rfe = rfe.fit_transform(X, y)

# RFE с кросс-валидацией для автоматического выбора количества
rfecv = RFECV(estimator, step=1, cv=5, scoring='accuracy')
X_rfecv = rfecv.fit_transform(X, y)
print(f"Оптимальное количество признаков: {rfecv.n_features_}")
```

### Отбор на основе важности модели
```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LassoCV

# Отбор на основе важности Random Forest
selector_rf = SelectFromModel(
    ExtraTreesClassifier(n_estimators=100, random_state=42),
    threshold='median'  # или конкретное значение
)
X_selected_rf = selector_rf.fit_transform(X, y)

# Отбор с помощью Lasso регуляризации
lasso_selector = SelectFromModel(LassoCV(cv=5, random_state=42))
X_selected_lasso = lasso_selector.fit_transform(X, y)
```

### Отбор по дисперсии
```python
from sklearn.feature_selection import VarianceThreshold

# Удаление признаков с низкой дисперсией
variance_selector = VarianceThreshold(threshold=0.01)
X_high_variance = variance_selector.fit_transform(X)
```

## 4. sklearn.model_selection - Валидация и подбор параметров

### Кросс-валидация
```python
from sklearn.model_selection import cross_val_score, StratifiedKFold, GroupKFold
from sklearn.ensemble import RandomForestClassifier

# Стратифицированная кросс-валидация
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(RandomForestClassifier(), X, y, cv=skf, scoring='roc_auc')

# Групповая кросс-валидация (когда есть группы)
gkf = GroupKFold(n_splits=5)
cv_scores_group = cross_val_score(RandomForestClassifier(), X, y, groups=groups, cv=gkf)
```

### Подбор гиперпараметров
```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

# Grid Search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
grid_search.fit(X, y)

# Randomized Search (быстрее для больших пространств параметров)
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': randint(2, 20)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=100,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42
)
```

### Разделение данных
```python
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

# Простое разделение
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Стратифицированное разделение с множественными разбиениями
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in sss.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
```

## 5. sklearn.pipeline - Пайплайны

### Простой пайплайн
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Создание пайплайна
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_classif, k=10)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Обучение и предсказание
pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)
```

### Комплексный пайплайн с разными типами данных
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Определение трансформаций для разных типов колонок
numeric_features = ['age', 'salary', 'experience']
categorical_features = ['department', 'education']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Объединение трансформаций
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Полный пайплайн
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selection', SelectFromModel(RandomForestClassifier())),
    ('classifier', RandomForestClassifier(random_state=42))
])
```

## 6. sklearn.metrics - Метрики качества

### Метрики для классификации
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, classification_report, confusion_matrix
)

# Базовые метрики
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# ROC-AUC для вероятностных предсказаний
roc_auc = roc_auc_score(y_true, y_prob[:, 1])  # для бинарной классификации
roc_auc_multi = roc_auc_score(y_true, y_prob, multi_class='ovr')  # мультиклассовая

# Логистическая потеря
logloss = log_loss(y_true, y_prob)
```

### Метрики для регрессии
```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_squared_log_error, mean_absolute_percentage_error
)

# Основные метрики регрессии
mse = mean_squared_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

# Специальные метрики
msle = mean_squared_log_error(y_true, y_pred)  # для положительных значений
mape = mean_absolute_percentage_error(y_true, y_pred)
```

## 7. sklearn.compose - Композиция трансформаций

### TransformedTargetRegressor
```python
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import RandomForestRegressor

# Трансформация целевой переменной
ttr = TransformedTargetRegressor(
    regressor=RandomForestRegressor(),
    transformer=PowerTransformer()
)
ttr.fit(X, y)
predictions = ttr.predict(X_test)
```

## 8. Практические советы для Kaggle

### Типичный пайплайн предобработки
```python
def create_kaggle_pipeline(numeric_features, categorical_features):
    """Создание типичного пайплайна для Kaggle соревнований"""
    
    # Числовые признаки
    numeric_transformer = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', RobustScaler()),
        ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False))
    ])
    
    # Категориальные признаки  
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Объединение
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    return Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', SelectFromModel(
            ExtraTreesClassifier(n_estimators=100), 
            threshold='median'
        )),
        ('classifier', RandomForestClassifier(n_estimators=300, random_state=42))
    ])
```

### Поиск оптимальных параметров
```python
def optimize_pipeline(pipeline, X, y):
    """Оптимизация пайплайна для Kaggle"""
    
    param_grid = {
        'preprocessor__num__imputer__n_neighbors': [3, 5, 7],
        'preprocessor__num__poly__degree': [1, 2],
        'feature_selection__threshold': ['mean', 'median', '0.75*mean'],
        'classifier__n_estimators': [200, 300, 500],
        'classifier__max_depth': [10, 15, None],
        'classifier__min_samples_split': [2, 5, 10]
    }
    
    # Используем Randomized Search для экономии времени
    search = RandomizedSearchCV(
        pipeline,
        param_grid,
        n_iter=50,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42
    )
    
    return search.fit(X, y)
```

### Обработка выбросов
```python
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope

def detect_outliers(X, contamination=0.1):
    """Детекция выбросов несколькими методами"""
    
    # Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outliers_iso = iso_forest.fit_predict(X)
    
    # Elliptic Envelope
    elliptic = EllipticEnvelope(contamination=contamination, random_state=42)
    outliers_elliptic = elliptic.fit_predict(X)
    
    # Объединение результатов (консенсус)
    outliers_consensus = (outliers_iso == -1) | (outliers_elliptic == -1)
    
    return outliers_consensus
```

## 9. Специфические техники для разных типов соревнований

### Time Series Competition
```python
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LaggedFeatures
import pandas as pd

def create_time_series_features(df, target_col, date_col):
    """Создание признаков для временных рядов"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    
    # Лаговые признаки
    for lag in [1, 2, 3, 7, 14, 30]:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    # Скользящие средние
    for window in [3, 7, 14, 30]:
        df[f'{target_col}_ma_{window}'] = df[target_col].rolling(window).mean()
        df[f'{target_col}_std_{window}'] = df[target_col].rolling(window).std()
    
    # Временные признаки
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['month'] = df[date_col].dt.month
    df['quarter'] = df[date_col].dt.quarter
    df['is_weekend'] = (df[date_col].dt.dayofweek >= 5).astype(int)
    
    # Экспоненциальное сглаживание
    df[f'{target_col}_ewm_7'] = df[target_col].ewm(span=7).mean()
    df[f'{target_col}_ewm_30'] = df[target_col].ewm(span=30).mean()
    
    return df

# Кросс-валидация для временных рядов
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
# Пайплайн для временных рядов
time_series_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='forward_fill')),  # forward fill для временных рядов
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(n_estimators=300))
])
```

### Text Classification / NLP
```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

def create_text_pipeline():
    """Пайплайн для текстовых данных"""
    
    # TF-IDF признаки
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2,
        max_df=0.95,
        lowercase=True,
        strip_accents='ascii'
    )
    
    # Снижение размерности для текста
    svd = TruncatedSVD(n_components=300, random_state=42)
    
    # Полный пайплайн
    text_pipeline = Pipeline([
        ('tfidf', tfidf),
        ('svd', svd),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    
    return text_pipeline

# Комбинирование текстовых и числовых признаков
def create_mixed_text_numeric_pipeline(text_cols, numeric_cols):
    """Пайплайн для смешанных данных (текст + числа)"""
    
    # Обработка текста
    text_transformer = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('svd', TruncatedSVD(n_components=100))
    ])
    
    # Обработка числовых данных
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Объединение
    preprocessor = ColumnTransformer([
        ('text', text_transformer, text_cols),
        ('num', numeric_transformer, numeric_cols)
    ])
    
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

# Продвинутые текстовые признаки
def extract_text_features(texts):
    """Извлечение дополнительных признаков из текста"""
    features = pd.DataFrame()
    
    features['text_length'] = texts.str.len()
    features['word_count'] = texts.str.split().str.len()
    features['char_count'] = texts.str.replace(' ', '').str.len()
    features['avg_word_length'] = features['char_count'] / features['word_count']
    features['sentence_count'] = texts.str.count('\.')
    features['exclamation_count'] = texts.str.count('!')
    features['question_count'] = texts.str.count('\?')
    features['uppercase_ratio'] = texts.str.count('[A-Z]') / features['char_count']
    features['digit_ratio'] = texts.str.count('\d') / features['char_count']
    
    return features
```

### Image Classification (с предобученными признаками)
```python
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

def create_image_pipeline():
    """Пайплайн для работы с признаками изображений"""
    
    # Обычно признаки уже извлечены CNN (например, ResNet features)
    image_pipeline = Pipeline([
        ('normalizer', normalize),  # L2 нормализация для CNN признаков
        ('pca', PCA(n_components=512)),  # Снижение размерности
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    
    return image_pipeline

def create_image_features_with_clustering(X_features):
    """Создание дополнительных признаков через кластеризацию"""
    
    # K-means кластеризация для создания новых признаков
    kmeans = KMeans(n_clusters=50, random_state=42)
    cluster_labels = kmeans.fit_predict(X_features)
    
    # Расстояния до центроидов как признаки
    distances = kmeans.transform(X_features)
    
    # Объединение оригинальных признаков с новыми
    enhanced_features = np.column_stack([
        X_features,
        cluster_labels.reshape(-1, 1),
        distances
    ])
    
    return enhanced_features
```

### Tabular Data с продвинутыми техниками
```python
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVC
import lightgbm as lgb

def create_advanced_tabular_pipeline():
    """Продвинутый пайплайн для табличных данных"""
    
    # Базовые модели для стэкинга
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=300, random_state=42)),
        ('et', ExtraTreesClassifier(n_estimators=300, random_state=42)),
        ('lgb', lgb.LGBMClassifier(random_state=42, verbose=-1))
    ]
    
    # Мета-модель
    meta_model = LogisticRegression(max_iter=1000)
    
    # Stacking
    stacking_classifier = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        stack_method='predict_proba'
    )
    
    return stacking_classifier

def create_feature_engineering_pipeline():
    """Автоматическое создание признаков"""
    
    class FeatureEngineer(BaseEstimator, TransformerMixin):
        def __init__(self, create_interactions=True, create_ratios=True):
            self.create_interactions = create_interactions
            self.create_ratios = create_ratios
            
        def fit(self, X, y=None):
            self.feature_names_ = X.columns if hasattr(X, 'columns') else None
            return self
            
        def transform(self, X):
            X_new = X.copy()
            
            if self.create_interactions and X.shape[1] < 20:  # только для небольших датасетов
                # Создание взаимодействий между числовыми признаками
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                for i, col1 in enumerate(numeric_cols):
                    for col2 in numeric_cols[i+1:]:
                        X_new[f'{col1}_mult_{col2}'] = X[col1] * X[col2]
                        X_new[f'{col1}_div_{col2}'] = X[col1] / (X[col2] + 1e-8)
            
            if self.create_ratios:
                # Создание относительных признаков
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    X_sum = X[numeric_cols].sum(axis=1)
                    for col in numeric_cols:
                        X_new[f'{col}_ratio'] = X[col] / (X_sum + 1e-8)
            
            return X_new
    
    return Pipeline([
        ('feature_eng', FeatureEngineer()),
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', RobustScaler()),
        ('feature_selection', SelectFromModel(
            ExtraTreesClassifier(n_estimators=100), 
            threshold='median'
        ))
    ])
```

### Продвинутые техники кросс-валидации
```python
from sklearn.model_selection import RepeatedStratifiedKFold, LeaveOneGroupOut

def advanced_cross_validation_strategies():
    """Продвинутые стратегии кросс-валидации"""
    
    # Повторная стратифицированная кросс-валидация
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    
    # Кросс-валидация с исключением групп (для данных с группировкой)
    logo = LeaveOneGroupOut()
    
    # Пользовательская кросс-валидация для временных рядов с gap
    class TimeSeriesKFoldWithGap:
        def __init__(self, n_splits=5, gap=0):
            self.n_splits = n_splits
            self.gap = gap
            
        def split(self, X, y=None, groups=None):
            n_samples = len(X)
            fold_size = n_samples // self.n_splits
            
            for i in range(self.n_splits):
                # Обучающая выборка
                train_end = (i + 1) * fold_size - self.gap
                train_indices = np.arange(0, train_end)
                
                # Валидационная выборка
                val_start = train_end + self.gap
                val_end = val_start + fold_size
                val_indices = np.arange(val_start, min(val_end, n_samples))
                
                if len(val_indices) > 0:
                    yield train_indices, val_indices
    
    return rskf, logo, TimeSeriesKFoldWithGap()

def nested_cross_validation(X, y, outer_cv=5, inner_cv=3):
    """Вложенная кросс-валидация для честной оценки"""
    
    # Внешняя кросс-валидация
    outer_scores = []
    skf_outer = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=42)
    
    for train_idx, test_idx in skf_outer.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Внутренняя кросс-валидация для подбора параметров
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, None]
        }
        
        inner_cv = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=inner_cv,
            scoring='roc_auc'
        )
        
        # Обучение на внутренней валидации
        grid_search.fit(X_train, y_train)
        
        # Оценка на внешней валидации
        best_model = grid_search.best_estimator_
        score = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
        outer_scores.append(score)
    
    return np.mean(outer_scores), np.std(outer_scores)
```

### Обработка несбалансированных данных
```python
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

def handle_imbalanced_data():
    """Методы работы с несбалансированными данными"""
    
    # SMOTE для создания синтетических примеров
    smote = SMOTE(random_state=42, k_neighbors=5)
    
    # ADASYN (адаптивная версия SMOTE)
    adasyn = ADASYN(random_state=42)
    
    # Комбинированный подход
    smote_tomek = SMOTETomek(random_state=42)
    
    # Пайплайн с ресэмплингом
    from imblearn.pipeline import Pipeline as ImbPipeline
    
    imbalanced_pipeline = ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('sampler', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(
            class_weight='balanced',  # дополнительная балансировка
            random_state=42
        ))
    ])
    
    return imbalanced_pipeline

def calculate_optimal_threshold(y_true, y_prob):
    """Поиск оптимального порога для несбалансированных данных"""
    from sklearn.metrics import precision_recall_curve
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold, f1_scores[optimal_idx]
```

### Ансамблирование и блендинг
```python
def create_ensemble_predictions(models, X_train, y_train, X_test, cv_folds=5):
    """Создание ансамбля через out-of-fold предсказания"""
    
    n_models = len(models)
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    
    # Матрицы для out-of-fold предсказаний
    oof_predictions = np.zeros((n_train, n_models))
    test_predictions = np.zeros((n_test, n_models))
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    for model_idx, (name, model) in enumerate(models):
        print(f"Training {name}...")
        
        test_fold_preds = np.zeros((n_test, cv_folds))
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Обучение модели
            model.fit(X_fold_train, y_fold_train)
            
            # Out-of-fold предсказания
            oof_predictions[val_idx, model_idx] = model.predict_proba(X_fold_val)[:, 1]
            
            # Предсказания на тесте
            test_fold_preds[:, fold_idx] = model.predict_proba(X_test)[:, 1]
        
        # Усреднение предсказаний на тесте
        test_predictions[:, model_idx] = test_fold_preds.mean(axis=1)
    
    return oof_predictions, test_predictions

def blend_predictions(oof_preds, test_preds, y_train, blend_method='linear'):
    """Блендинг предсказаний разных моделей"""
    
    if blend_method == 'linear':
        # Линейное блендинг через Ridge регрессию
        from sklearn.linear_model import Ridge
        blender = Ridge(alpha=1.0)
        blender.fit(oof_preds, y_train)
        
        final_test_preds = blender.predict(test_preds)
        weights = blender.coef_
        
    elif blend_method == 'rank_average':
        # Ранговое усреднение
        from scipy.stats import rankdata
        
        # Преобразование в ранги
        oof_ranks = np.column_stack([rankdata(oof_preds[:, i]) for i in range(oof_preds.shape[1])])
        test_ranks = np.column_stack([rankdata(test_preds[:, i]) for i in range(test_preds.shape[1])])
        
        # Простое усреднение рангов
        final_test_preds = test_ranks.mean(axis=1)
        weights = np.ones(oof_preds.shape[1]) / oof_preds.shape[1]
    
    return final_test_preds, weights
```

## Ключевые принципы для Kaggle:

1. **Всегда используйте пайплайны** - это предотвращает data leakage
2. **KNN Imputer часто работает лучше простых стратегий** для числовых данных
3. **Robust Scaler устойчив к выбросам** лучше Standard Scaler
4. **Feature selection критически важен** - больше признаков не всегда лучше
5. **Используйте стратифицированную кросс-валидацию** для несбалансированных классов
6. **Randomized Search экономит время** при подборе гиперпараметров
7. **Создавайте новые признаки** через полиномиальные преобразования и взаимодействия
8. **Для временных рядов используйте TimeSeriesSplit** и не забывайте про data leakage
9. **В NLP задачах TF-IDF + SVD** часто дает хорошую baseline
10. **Ансамблирование почти всегда улучшает результат** - используйте разные типы моделей
11. **Nested CV дает честную оценку** качества вашей модели
12. **Для несбалансированных данных** комбинируйте SMOTE с class_weight='balanced'