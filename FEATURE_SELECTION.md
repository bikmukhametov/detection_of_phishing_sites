# Отбор признаков (Feature Selection)

Этот модуль содержит реализацию трех алгоритмов отбора признаков для классификации фишинговых веб-сайтов.

## Структура модуля

### `src/feature_selection.py`
Содержит реализацию алгоритмов отбора признаков и оценки их качества.

**Основные компоненты:**

#### 1. **Алгоритм Add-Del (Sequential Addition-Deletion)**
```python
add_del_algorithm(X, y, feature_names, max_iterations=100, patience=5, test_size=0.2)
```

**Описание:** Комбинирует прямой отбор (forward selection) и обратное исключение (backward elimination).

**Алгоритм:**
1. Начинаем с пустого набора признаков J₀
2. **Фаза добавления (Add):** Итеративно добавляем признаки, которые минимизируют критерий Q
   - Q(J) = 1 - accuracy (ошибка на обучающей выборке)
   - Продолжаем, пока |J| < n
3. **Фаза удаления (Del):** Удаляем признаки, которые минимизируют Q
   - Продолжаем, пока |J| > 0
4. **Остановка:** Когда Q не улучшается в течение d итераций (patience)

**Параметры:**
- `X`: Матрица признаков (n_samples x n_features)
- `y`: Метки классов
- `feature_names`: Названия признаков
- `max_iterations`: Максимум итераций на фазу
- `patience`: Количество итераций без улучшения перед остановкой
- `test_size`: Доля тестовой выборки (default: 0.2)

**Возвращает:**
- Индексы выбранных признаков
- Словарь статистики алгоритма

---

#### 2. **Генетический алгоритм (Genetic Algorithm)**
```python
genetic_algorithm(X, y, feature_names, population_size=50, generations=50, 
                  mutation_rate=0.1, crossover_prob=0.8, test_size=0.2)
```

**Описание:** Эволюционный поиск оптимального набора признаков.

**Алгоритм:**
1. **Инициализация:** Создаем начальную популяцию из B случайных двоичных масок (хромосом)
2. **Оценка:** Оцениваем приспособленность каждой хромосомы (1/Q - штраф за количество признаков)
3. **Для каждого поколения:**
   - **Отбор (Selection):** Турнирный отбор - выбираем лучших особей
   - **Кроссовер (Crossover):** С вероятностью P_cross комбинируем родителей
     - Равномерный кроссовер (uniform crossover)
   - **Мутация (Mutation):** С вероятностью P_mut переворачиваем биты
   - **Замена поколения:** Создаем новое поколение из потомков

**Параметры:**
- `population_size`: Размер популяции
- `generations`: Количество поколений
- `mutation_rate`: Вероятность мутации для каждого признака
- `crossover_prob`: Вероятность кроссовера

**Возвращает:**
- Индексы выбранных признаков (лучшая найденная хромосома)
- Словарь статистики алгоритма

---

#### 3. **Случайный поиск с адаптацией (СПА — Stochastic Search with Adaptation, SPA)**
```python
stochastic_search_with_adaptation(X, y, feature_names, j0=1, T=30, r=10, h=0.05, test_size=0.2)
```

**Описание:** Случайный поиск с адаптацией вероятностей выбора признаков.

**Алгоритм (кратко):**
1. Инициализируем вероятности выбора признаков p_s = 1/n
2. Для каждой сложности j (размер подмножества) и каждой итерации t:
   - Генерируем r случайных наборов размера j по распределению p
   - Находим лучший (J_min) и худший (J_max) наборы по Q
   - Накладываем «наказание» на признаки из J_max: p_s := p_s - Δp_s, Δp_s = min(p_s, h)
   - На признаки из J_min добавляем долю накопленного наказания H: p_s := p_s + H / |J_min|
   - Нормализуем p
3. Возвращаем лучший найденный набор J*

**Параметры:**
- `j0`: минимальный размер подмножеств
- `T`: число итераций на каждую сложность
- `r`: число выборок на итерацию
- `h`: максимальная величина наказания для признака
- `test_size`: доля тестовой выборки

**Возвращает:**
- Индексы выбранных признаков
- Словарь статистики алгоритма

---

### `src/feature_analysis.py`
Содержит функции анализа и визуализации результатов отбора признаков.

**Функции:**

1. **`save_feature_selection_results(results, output_dir)`**
   - Сохраняет единый сводный отчет в Markdown (`feature_selection_summary.md`)
   - Не создает дублирующие `.csv`, `.json` или отдельные `*_features.txt` — отчёт и графики считаются основными артефактами

2. **`plot_algorithm_comparison(results, output_dir)`**
   - Создает графики сравнения метрик между алгоритмами
   - Включает: Accuracy, Precision, Recall, F1, ROC-AUC

3. **`plot_quality_vs_feature_count(results, output_dir)`**
   - График зависимости Q (критерия ошибки) от количества выбранных признаков
   - Соответствует примеру из задания

4. **`plot_convergence_curves(results, output_dir)`**
   - Кривые сходимости для каждого алгоритма
   - Показывает Q по итерациям/поколениям

5. **`plot_feature_count_progression(results, output_dir)`**
   - График изменения количества признаков по итерациям
   - Помогает отследить эволюцию процесса поиска

6. **`create_summary_markdown(results, output_dir, all_feature_names)`**
   - Создает подробный отчет в формате Markdown
   - Включает таблицу сравнения и описание выбранных признаков для каждого алгоритма

---

## Использование

### Быстрый тест (на выборке)
```bash
python test_feature_selection.py
```

Результаты сохраняются в `outputs/feature_selection_test/`

### Полный анализ (на всех данных)
```bash
python main.py
```

Результаты сохраняются в `outputs/feature_selection/`

### Использование в коде
```python
from src.feature_selection import (
   add_del_algorithm,
   genetic_algorithm,
   stochastic_search_with_adaptation,
   evaluate_feature_set
)
from src.feature_analysis import (
    save_feature_selection_results,
    plot_algorithm_comparison,
    plot_quality_vs_feature_count,
    plot_convergence_curves,
    plot_feature_count_progression,
    create_summary_markdown
)

# Загрузить данные
from src.data_loader import load_dataset
from src.preprocess import preprocess_features

df, feature_df, label_series = load_dataset("data.csv", label_col="Result")
X_scaled, _ = preprocess_features(feature_df)

# Запустить алгоритмы
selected1, stats1 = add_del_algorithm(X_scaled, label_series.values, list(feature_df.columns))
selected2, stats2 = genetic_algorithm(X_scaled, label_series.values, list(feature_df.columns))
selected3, stats3 = stochastic_search_with_adaptation(X_scaled, label_series.values, list(feature_df.columns))

# Сохранить и визуализировать
results = {
    'Add-Del': {'selected_features': selected1, ...},
    'Genetic Algorithm': {'selected_features': selected2, ...},
    'SSA': {'selected_features': selected3, ...}
}
save_feature_selection_results(results, Path("outputs"))
plot_algorithm_comparison(results, Path("outputs"))
```

---

## Выходные данные

### Файлы, создаваемые в `outputs/feature_selection/`:

1. **feature_selection_summary.md** - единый отчёт в Markdown (сводная таблица + подробности)
2. **algorithms_metrics_comparison.png** - Сравнение метрик
3. **roc_auc_and_feature_count.png** - ROC-AUC и количество признаков
4. **quality_vs_feature_count.png** - График Q от количества признаков (как в задании)
5. **convergence_curves.png** - Кривые сходимости
6. **feature_count_progression.png** - Изменение количества признаков

---

## Метрики качества

Для каждого набора признаков рассчитываются:

- **Accuracy** = (TP + TN) / (TP + TN + FP + FN)
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1 Score** = 2 * (Precision * Recall) / (Precision + Recall)
- **ROC-AUC** = Площадь под ROC-кривой
- **Q (Error Rate)** = 1 - Accuracy

где:
- TP = True Positives (правильно классифицированы как фишинг)
- TN = True Negatives (правильно классифицированы как легитимные)
- FP = False Positives (ошибочно классифицированы как фишинг)
- FN = False Negatives (ошибочно классифицированы как легитимные)

---

## Примеры результатов

### Пример вывода на тестовой выборке (500 образцов):

```
Testing Add-Del...
Add-Del: 2 features, F1=0.9189

Testing Genetic Algorithm...
Genetic Algorithm: 21 features, F1=0.9630

Testing SPA...
SPA: 21 features, F1=0.9541
```

### Сравнение алгоритмов:

| Алгоритм | Признаков | Accuracy | Precision | Recall | F1 | ROC-AUC |
|----------|-----------|----------|-----------|--------|----|---------| 
| Add-Del | 2 | 0.9100 | 0.8947 | 0.9444 | 0.9189 | 0.9257 |
| Genetic Algorithm | 21 | 0.9600 | 0.9630 | 0.9630 | 0.9630 | 0.9952 |
| Stochastic Search (SPA) | 21 | 0.9500 | 0.9455 | 0.9630 | 0.9541 | 0.9867 |

**Выводы:**
- **Add-Del** быстро находит минимальный набор (2 признака), но качество ниже
-- **Genetic Algorithm** и **SPA** находят более сложные наборы с лучшим качеством
- Генетический алгоритм показал лучший ROC-AUC (0.9952)
- Все алгоритмы демонстрируют хорошую обобщающую способность (F1 > 0.91)

---

## Параметры запуска

Для полного анализа рекомендуется использовать:

### Add-Del Algorithm:
```python
add_del_algorithm(X_scaled, y, feature_names,
                  max_iterations=100,
                  patience=10,
                  test_size=0.2)
```

### Genetic Algorithm:
```python
genetic_algorithm(X_scaled, y, feature_names,
                 population_size=50,
                 generations=50,
                 mutation_rate=0.1,
                 crossover_prob=0.8,
                 test_size=0.2)
```

### Stochastic Search (SPA):
```python
stochastic_search_with_adaptation(X_scaled, y, feature_names,
                                    j0=1,
                                    T=100,
                                    r=5,
                                    h=0.05,
                                    test_size=0.2)
```

---

## Ссылки на теорию

- **Add-Del Algorithm**: Последовательный отбор признаков (Sequential Feature Selection)
- **Genetic Algorithm**: Эволюционные алгоритмы оптимизации
-- **SPA**: Случайный поиск в пространстве подмножеств признаков с адаптацией вероятностей (Random Search with Adaptation)

Все алгоритмы используют критерий качества Q (ошибка классификации) и логистическую регрессию как базовый классификатор.
