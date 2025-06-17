import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class AnomalyDetector:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.pca = None
        
    def prepare_data(self, X_train, X_test, use_pca=True, n_components=0.95):
        """Подготовка данных с нормализацией и PCA"""
        
        # Нормализация - используем RobustScaler для устойчивости к выбросам
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if use_pca and X_train.shape[1] > 10:
            # PCA для снижения размерности
            self.pca = PCA(n_components=n_components)
            X_train_scaled = self.pca.fit_transform(X_train_scaled)
            X_test_scaled = self.pca.transform(X_test_scaled)
            
            print(f"PCA: {X_train.shape[1]} -> {X_train_scaled.shape[1]} features")
        
        return X_train_scaled, X_test_scaled
    
    def train_isolation_forest(self, X_train, contamination=0.1):
        """Обучение Isolation Forest"""
        
        # Подбор параметров
        models = {}
        
        # Разные варианты contamination
        for cont in [0.05, 0.1, 0.15, 0.2]:
            model = IsolationForest(
                contamination=cont,
                random_state=42,
                n_estimators=200,
                max_samples='auto',
                bootstrap=True
            )
            model.fit(X_train)
            models[f'IF_{cont}'] = model
        
        # Модель с auto contamination
        model_auto = IsolationForest(
            contamination='auto',
            random_state=42,
            n_estimators=200,
            max_samples='auto',
            bootstrap=True
        )
        model_auto.fit(X_train)
        models['IF_auto'] = model_auto
        
        self.models.update(models)
        return models
    
    def train_one_class_svm(self, X_train):
        """Обучение One-Class SVM"""
        
        models = {}
        
        # Разные ядра и параметры
        configs = [
            {'kernel': 'rbf', 'gamma': 'scale', 'nu': 0.05},
            {'kernel': 'rbf', 'gamma': 'scale', 'nu': 0.1},
            {'kernel': 'rbf', 'gamma': 'auto', 'nu': 0.1},
            {'kernel': 'linear', 'nu': 0.1},
            {'kernel': 'poly', 'degree': 3, 'nu': 0.1},
        ]
        
        for i, config in enumerate(configs):
            model = OneClassSVM(**config)
            model.fit(X_train)
            models[f'OCSVM_{i}'] = model
        
        self.models.update(models)
        return models
    
    def ensemble_predict(self, X_test, voting='soft'):
        """Ансамблевое предсказание"""
        
        predictions = {}
        scores = {}
        
        for name, model in self.models.items():
            # Предсказания (-1 для аномалий, 1 для нормальных)
            pred = model.predict(X_test)
            predictions[name] = pred
            
            # Скоры (для Isolation Forest - аномалии имеют низкие скоры)
            if hasattr(model, 'decision_function'):
                score = model.decision_function(X_test)
                scores[name] = score
            elif hasattr(model, 'score_samples'):
                score = model.score_samples(X_test)
                scores[name] = score
        
        # Объединение предсказаний
        pred_df = pd.DataFrame(predictions)
        
        if voting == 'hard':
            # Жесткое голосование
            ensemble_pred = pred_df.mode(axis=1)[0]
        else:
            # Мягкое голосование на основе скоров
            if scores:
                score_df = pd.DataFrame(scores)
                # Нормализуем скоры
                score_df_norm = (score_df - score_df.min()) / (score_df.max() - score_df.min())
                # Средний скор
                avg_score = score_df_norm.mean(axis=1)
                # Определяем порог
                threshold = np.percentile(avg_score, 10)  # 10% как аномалии
                ensemble_pred = np.where(avg_score < threshold, -1, 1)
            else:
                ensemble_pred = pred_df.mode(axis=1)[0]
        
        return ensemble_pred, predictions, scores
    
    def evaluate_models(self, X_test, y_true=None):
        """Оценка моделей"""
        
        results = {}
        
        for name, model in self.models.items():
            pred = model.predict(X_test)
            # Конвертируем -1/1 в 0/1 (0 - нормальный, 1 - аномалия)
            pred_binary = np.where(pred == -1, 1, 0)
            
            if y_true is not None:
                auc = roc_auc_score(y_true, pred_binary)
                results[name] = {'predictions': pred_binary, 'auc': auc}
            else:
                results[name] = {'predictions': pred_binary}
        
        return results

def main():
    # Загрузка данных
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    # Выделение признаков
    feature_cols = [col for col in train_df.columns if col not in ['id', 'is_anomaly', 'target']]
    
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    
    # Если есть метки в обучающей выборке
    y_train = train_df['is_anomaly'].values if 'is_anomaly' in train_df.columns else None
    
    # Инициализация детектора
    detector = AnomalyDetector()
    
    # Подготовка данных
    X_train_prep, X_test_prep = detector.prepare_data(X_train, X_test)
    
    # Для обучения используем только нормальные данные (если известны метки)
    if y_train is not None:
        normal_mask = y_train == 0
        X_train_normal = X_train_prep[normal_mask]
        print(f"Using {len(X_train_normal)} normal samples for training")
    else:
        X_train_normal = X_train_prep
        print(f"Using all {len(X_train_normal)} samples for training")
    
    # Обучение моделей
    print("Training Isolation Forest models...")
    if_models = detector.train_isolation_forest(X_train_normal)
    
    print("Training One-Class SVM models...")
    svm_models = detector.train_one_class_svm(X_train_normal)
    
    # Предсказания
    print("Making predictions...")
    ensemble_pred, all_preds, all_scores = detector.ensemble_predict(X_test_prep)
    
    # Конвертируем в бинарный формат (0 - нормальный, 1 - аномалия)
    ensemble_binary = np.where(ensemble_pred == -1, 1, 0)
    
    # Оценка отдельных моделей
    if y_train is not None:
        # Создаем валидационную выборку
        X_val = X_train_prep[~normal_mask] if len(X_train_prep[~normal_mask]) > 0 else X_train_prep[:100]
        y_val = y_train[~normal_mask] if len(y_train[~normal_mask]) > 0 else np.zeros(100)
        
        val_results = detector.evaluate_models(X_val, y_val)
        
        print("\nModel Performance on Validation:")
        for name, result in val_results.items():
            if 'auc' in result:
                print(f"{name}: AUC = {result['auc']:.4f}")
    
    # Создание submission файла
    submission = pd.DataFrame({
        'id': test_df['id'] if 'id' in test_df.columns else range(len(test_df)),
        'is_anomaly': ensemble_binary
    })
    
    # Дополнительные файлы с предсказаниями отдельных моделей
    detailed_results = pd.DataFrame(all_preds)
    detailed_results['id'] = test_df['id'] if 'id' in test_df.columns else range(len(test_df))
    detailed_results['ensemble'] = ensemble_binary
    
    # Сохранение результатов
    submission.to_csv('anomaly_submission_ensemble.csv', index=False)
    detailed_results.to_csv('detailed_predictions.csv', index=False)
    
    print(f"\nEnsemble Results:")
    print(f"Total samples: {len(ensemble_binary)}")
    print(f"Detected anomalies: {np.sum(ensemble_binary)} ({np.mean(ensemble_binary)*100:.1f}%)")
    
    # Статистика по моделям
    pred_df = pd.DataFrame(all_preds)
    pred_binary = pred_df.applymap(lambda x: 1 if x == -1 else 0)
    
    print(f"\nIndividual Model Stats:")
    for col in pred_binary.columns:
        anomaly_rate = pred_binary[col].mean() * 100
        print(f"{col}: {anomaly_rate:.1f}% anomalies")

# Дополнительная функция для визуализации (если нужно)
def visualize_anomalies(X, predictions, feature_names=None, save_path='anomaly_viz.png'):
    """Визуализация аномалий в 2D проекции"""
    
    # PCA для визуализации
    pca_viz = PCA(n_components=2)
    X_2d = pca_viz.fit_transform(X)
    
    plt.figure(figsize=(12, 8))
    
    # Нормальные точки
    normal_mask = predictions == 0
    plt.scatter(X_2d[normal_mask, 0], X_2d[normal_mask, 1], 
                c='blue', alpha=0.6, label='Normal', s=50)
    
    # Аномалии
    anomaly_mask = predictions == 1
    plt.scatter(X_2d[anomaly_mask, 0], X_2d[anomaly_mask, 1], 
                c='red', alpha=0.8, label='Anomaly', s=100, marker='x')
    
    plt.xlabel(f'PC1 ({pca_viz.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca_viz.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('Anomaly Detection Results (PCA Projection)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()