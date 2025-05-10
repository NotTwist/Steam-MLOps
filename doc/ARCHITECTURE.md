## Описание проекта
**Источник данных:** Steam Games Dataset (games.json) из Kaggle  
**Целевая переменная:** price (прогноз цены игры)  
**Валидация данных:**  
  - Проверка на пропуски (max_missing), дубликаты (max_duplicates), выбросы (max_outliers)  
  - Валидация батчей в dataset_utils.py/validate_batch()  
**Хранилище артефактов:**  
  - Модели: storage/results/model_results_<timestamp>/  
  - Отчеты EDA: storage/results/eda/  
  - Метрики: storage/results/reports/  

## Архитектура модели
**Feature Processing:**  
- Категориальные признаки (tags, genres, categories) кодируются через MultiLabelBinarizer  
- Игнорируемые колонки: app_id, name и др. (см. config.yaml)  

**Настройки гиперпараметров:**  
- Для каждой модели задается сетка параметров (например, KNeighborsRegressor с n_neighbors=[3,5,7,9])  
- Оптимизация через GridSearchCV с кросс-валидацией (5 фолдов)  

**Отбор модели:**  
- Лучшая модель выбирается по минимальному MAE  

## Design Decisions
**Model Choices:**  
- Линейные модели (LinearRegression) для базовой оценки  
- Непараметрические методы (KNN, DecisionTree) для улавливания нелинейностей  

**Incremental Training:**  
- Обучение на новых батчах без перезаписи предыдущих моделей  
- Сбор уникальных категорий из всех батчей для корректного кодирования  

**Data Drift Handling:**  
- Мониторинг дрейфа через KS-тест (detect_data_drift())  
- Отчеты сохраняются в YAML (data_drift_report_*.yaml)  