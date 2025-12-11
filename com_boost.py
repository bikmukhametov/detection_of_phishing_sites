import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from copy import deepcopy
import warnings

class ComBoost(BaseEstimator, ClassifierMixin):
    """Реализация алгоритма Committee Boosting (ComBoost)"""
    
    def __init__(self, base_estimator=None, T=10, l0=0.1, l1=0.3, 
                 l2=0.7, delta_l=0.1, early_stopping=True, 
                 min_improvement=0.001, verbose=False):
        """
        Параметры:
        ----------
        base_estimator : object
            Базовая модель для бустинга (по умолчанию DecisionTreeClassifier(max_depth=3))
        T : int
            Максимальное количество моделей в комитете
        l0 : float
            Нижняя граница отступов (процент от выборки)
        l1 : float
            Начальный размер подмножества (процент от выборки)
        l2 : float
            Конечный размер подмножества (процент от выборки)
        delta_l : float
            Шаг увеличения размера подмножества
        early_stopping : bool
            Останавливать ли обучение при отсутствии улучшений
        min_improvement : float
            Минимальное улучшение для продолжения обучения
        verbose : bool
            Вывод информации о процессе обучения
        """
        self.base_estimator = base_estimator if base_estimator else DecisionTreeClassifier(max_depth=3)
        self.T = T
        self.l0 = l0
        self.l1 = l1
        self.l2 = l2
        self.delta_l = delta_l
        self.early_stopping = early_stopping
        self.min_improvement = min_improvement
        self.verbose = verbose
        
        # Для хранения истории
        self.models = []
        self.margins_history = []
        self.scores_history = []
        self.best_k_history = []
        
    def _quality_function(self, model, X, y):
        """Функция качества Q(b, X) - accuracy"""
        y_pred = model.predict(X)
        return accuracy_score(y, y_pred)
    
    def fit(self, X, y):
        """
        Обучение комитета по алгоритму ComBoost
        
        Параметры:
        ----------
        X : array-like, shape (n_samples, n_features)
            Обучающие данные
        y : array-like, shape (n_samples,)
            Целевые значения
            
        Возвращает:
        -----------
        self : object
        """
        X = np.array(X)
        y = np.array(y)
        n_samples = len(X)
        
        # Преобразуем проценты в абсолютные значения
        l0_abs = max(1, int(self.l0 * n_samples))
        l1_abs = max(l0_abs + 1, int(self.l1 * n_samples))
        l2_abs = min(n_samples, int(self.l2 * n_samples))
        delta_l_abs = max(1, int(self.delta_l * n_samples))
        
        if l2_abs <= l1_abs:
            l2_abs = min(n_samples, l1_abs + delta_l_abs)
        
        # 1. Обучение первой модели на всей выборке
        if self.verbose:
            print("Шаг 1: Обучение первой модели на всей выборке")
        
        b1 = deepcopy(self.base_estimator)
        b1.fit(X, y)
        self.models.append(b1)
        
        # Инициализация отступов
        margins = y * b1.predict(X)
        self.margins_history.append(margins.copy())
        
        # Вычисление качества первой модели
        initial_score = self._quality_function(b1, X, y)
        self.scores_history.append(initial_score)
        
        if self.verbose:
            print(f"  Качество первой модели: {initial_score:.4f}")
        
        # Упорядочивание индексов по возрастанию отступов
        indices = np.argsort(margins)
        
        # Основной цикл бустинга
        prev_best_score = initial_score
        
        for t in range(1, self.T):
            if self.verbose:
                print(f"\nИтерация {t+1}/{self.T}")
            
            best_model = None
            best_score = -np.inf
            best_k = None
            
            # 3-5. Перебор различных подмножеств данных
            k_values = range(l1_abs, l2_abs + 1, delta_l_abs)
            
            for k in k_values:
                if k <= l0_abs:
                    continue
                    
                # 4. Формирование подмножества U
                U_indices = indices[l0_abs:k]
                
                if len(U_indices) == 0:
                    continue
                
                # 5. Обучение модели на подмножестве U
                try:
                    b_tk = deepcopy(self.base_estimator)
                    b_tk.fit(X[U_indices], y[U_indices])
                    
                    # Вычисление качества на подмножестве U
                    score = self._quality_function(b_tk, X[U_indices], y[U_indices])
                    
                    if score > best_score:
                        best_score = score
                        best_model = b_tk
                        best_k = k
                        
                except Exception as e:
                    if self.verbose:
                        print(f"  Ошибка при обучении с k={k}: {e}")
                    continue
            
            # 6. Выбор наилучшей модели
            if best_model is None:
                if self.verbose:
                    print(f"  Не удалось обучить модель на итерации {t+1}")
                break
            
            self.models.append(best_model)
            self.best_k_history.append(best_k)
            
            # 7. Обновление отступов
            y_pred = best_model.predict(X)
            margins += y * y_pred
            self.margins_history.append(margins.copy())
            
            # 8. Упорядочивание выборки по возрастанию отступов
            indices = np.argsort(margins)
            
            # Вычисление общего качества комитета
            current_score = self._quality_function(self, X, y)
            self.scores_history.append(current_score)
            
            if self.verbose:
                print(f"  Размер подмножества k: {best_k}")
                print(f"  Качество на подмножестве: {best_score:.4f}")
                print(f"  Общее качество комитета: {current_score:.4f}")
                print(f"  Средний отступ: {np.mean(margins):.4f}")
                print(f"  Минимальный отступ: {np.min(margins):.4f}")
            
            # 9-10. Проверка условия остановки
            if self.early_stopping and t > 0:
                improvement = current_score - prev_best_score
                
                if improvement < self.min_improvement:
                    if self.verbose:
                        print(f"\nРанняя остановка: улучшение ({improvement:.4f}) меньше порога ({self.min_improvement})")
                    break
            
            prev_best_score = current_score
        
        if self.verbose:
            print(f"\nОбучение завершено. Обучено {len(self.models)} моделей")
            print(f"Финальное качество: {self.scores_history[-1]:.4f}")
        
        return self
    
    def predict(self, X):
        """Предсказание методом голосования комитета"""
        if not self.models:
            raise ValueError("Модель не обучена. Вызовите fit() сначала.")
        
        X = np.array(X)
        
        # Собираем предсказания всех моделей
        predictions = np.zeros((len(X), len(self.models)))
        
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)
        
        # Голосование большинством
        final_predictions = np.zeros(len(X))
        
        for i in range(len(X)):
            votes = predictions[i, :]
            # Находим наиболее частый класс
            unique, counts = np.unique(votes, return_counts=True)
            final_predictions[i] = unique[np.argmax(counts)]
        
        return final_predictions
    
    def visualize_margins(self, figsize=(15, 10)):
        """
        Визуализация изменения отступов в процессе обучения
        
        Параметры:
        ----------
        figsize : tuple
            Размер фигуры
        """
        if not self.margins_history:
            print("Модель не обучена. Нет данных для визуализации.")
            return
        
        margins_array = np.array(self.margins_history)
        n_iterations = margins_array.shape[0]
        n_samples = margins_array.shape[1]
        
        fig = plt.figure(figsize=figsize)
        
        # 1. График распределения отступов по итерациям
        ax1 = plt.subplot(2, 2, 1)
        for i in range(min(5, n_iterations)):
            ax1.hist(margins_array[i], alpha=0.5, bins=30, 
                    label=f'Итерация {i+1}', density=True)
        ax1.set_xlabel('Отступы')
        ax1.set_ylabel('Плотность')
        ax1.set_title('Распределение отступов по итерациям')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. График среднего и минимального отступа
        ax2 = plt.subplot(2, 2, 2)
        mean_margins = np.mean(margins_array, axis=1)
        min_margins = np.min(margins_array, axis=1)
        
        iterations = range(1, n_iterations + 1)
        ax2.plot(iterations, mean_margins, 'b-o', linewidth=2, markersize=6, label='Средний отступ')
        ax2.plot(iterations, min_margins, 'r--s', linewidth=2, markersize=6, label='Минимальный отступ')
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Номер итерации')
        ax2.set_ylabel('Отступ')
        ax2.set_title('Динамика отступов в процессе обучения')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Тепловая карта отступов
        ax3 = plt.subplot(2, 2, 3)
        # Сортируем примеры по финальным отступам для лучшей визуализации
        sorted_indices = np.argsort(margins_array[-1])
        sorted_margins = margins_array[:, sorted_indices]
        
        im = ax3.imshow(sorted_margins.T, aspect='auto', cmap='RdYlBu',
                       extent=[1, n_iterations, 0, n_samples])
        ax3.set_xlabel('Номер итерации')
        ax3.set_ylabel('Примеры (отсортированы)')
        ax3.set_title('Тепловая карта отступов по примерам')
        plt.colorbar(im, ax=ax3, label='Значение отступа')
        
        # 4. График качества и выбранных k
        ax4 = plt.subplot(2, 2, 4)
        ax4.plot(iterations, self.scores_history, 'g-o', linewidth=2, markersize=6, label='Качество')
        ax4.set_xlabel('Номер итерации')
        ax4.set_ylabel('Accuracy', color='g')
        ax4.set_title('Качество комитета и размер подмножества')
        ax4.tick_params(axis='y', labelcolor='g')
        ax4.grid(True, alpha=0.3)
        
        # Вторначная ось для размера подмножества k
        if self.best_k_history:
            ax4_2 = ax4.twinx()
            ax4_2.plot(range(2, len(self.best_k_history) + 2), self.best_k_history, 
                      'm-s', linewidth=1, markersize=4, label='Размер k', alpha=0.7)
            ax4_2.set_ylabel('Размер подмножества k', color='m')
            ax4_2.tick_params(axis='y', labelcolor='m')
            ax4_2.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.show()
        
        # Дополнительная информация
        print("\nСтатистика по отступам:")
        print(f"  Начальный средний отступ: {mean_margins[0]:.4f}")
        print(f"  Финальный средний отступ: {mean_margins[-1]:.4f}")
        print(f"  Изменение среднего отступа: {mean_margins[-1] - mean_margins[0]:.4f}")
        print(f"  Начальный минимальный отступ: {min_margins[0]:.4f}")
        print(f"  Финальный минимальный отступ: {min_margins[-1]:.4f}")
        
        # Количество отрицательных отступов
        negative_margins_final = np.sum(margins_array[-1] < 0)
        print(f"  Примеров с отрицательными отступами: {negative_margins_final}/{n_samples}")

# =============================================================================
# АДАПТАЦИЯ ПОД ВАШ ДАТАСЕТ
# =============================================================================

# =============================================================================
# АДАПТАЦИЯ ПОД ВАШ ДАТАСЕТ
# =============================================================================

def prepare_data_from_your_dataset():
    """
    Функция для подготовки данных из вашего датасета.
    Замените эту функцию на загрузку ваших реальных данных.
    """
    print("=" * 60)
    print("ПОДГОТОВКА ДАННЫХ ИЗ ВАШЕГО ДАТАСЕТА")
    print("=" * 60)
    
    # ВАЖНО: Здесь должна быть ваша реальная загрузка данных
    # Пример структуры (замените на ваши данные):
    
    # Вариант 1: Если данные в CSV файле
    df = pd.read_csv('Phishing_Websites_Data.csv')
    X = df.drop('Result', axis=1).values
    y = df['Result'].values
    
    
    
    from sklearn.model_selection import train_test_split
    
    # Преобразуем классы 0,1 в -1,1
    y = 2 * y - 1  # 0 -> -1, 1 -> 1
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")
    print(f"Количество признаков: {X_train.shape[1]}")
    
    # Подсчет количества каждого класса (работает с -1 и 1)
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    print(f"Количество классов: {len(unique_classes)}")
    print(f"Классы в данных: {unique_classes}")
    print(f"Распределение классов в обучающей выборке: ", end="")
    for cls, count in zip(unique_classes, class_counts):
        print(f"класс {cls}: {count} ({count/len(y_train):.1%})", end=", ")
    print()
    
    return X_train, X_test, y_train, y_test


def run_comboost_on_your_data():
    """Запуск ComBoost на ваших данных"""
    print("\n" + "=" * 60)
    print("ЗАПУСК COMBOOST НА ВАШИХ ДАННЫХ")
    print("=" * 60)
    
    # 1. Загружаем ваши данные
    X_train, X_test, y_train, y_test = prepare_data_from_your_dataset()
    
    # 2. Создаем и обучаем ComBoost
    print("\nСоздание модели ComBoost...")
    comboost = ComBoost(
        base_estimator=DecisionTreeClassifier(max_depth=4, random_state=42),
        T=12,  # Максимальное количество моделей
        l0=0.1,   # Нижняя граница - 10% примеров
        l1=0.3,   # Начальный размер - 30% примеров
        l2=0.8,   # Максимальный размер - 80% примеров
        delta_l=0.1,  # Шаг увеличения размера
        early_stopping=True,
        min_improvement=0.0005,
        verbose=True  # Включаем подробный вывод
    )
    
    print("Начало обучения...")
    comboost.fit(X_train, y_train)
    
    # 3. Оценка модели
    print("\n" + "=" * 60)
    print("ОЦЕНКА МОДЕЛИ")
    print("=" * 60)
    
    # На обучающей выборке
    y_train_pred = comboost.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # На тестовой выборке
    y_test_pred = comboost.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"Количество обученных моделей в комитете: {len(comboost.models)}")
    print(f"Точность на обучающей выборке: {train_accuracy:.4f}")
    print(f"Точность на тестовой выборке: {test_accuracy:.4f}")
    print(f"История качества: {[round(s, 4) for s in comboost.scores_history]}")
    
    # 4. Визуализация
    print("\n" + "=" * 60)
    print("ВИЗУАЛИЗАЦИЯ ПРОЦЕССА ОБУЧЕНИЯ")
    print("=" * 60)
    comboost.visualize_margins()
    
    # 5. Дополнительный анализ
    print("\n" + "=" * 60)
    print("ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ")
    print("=" * 60)
    
    # Анализ важности признаков (среднее по всем деревьям)
    if hasattr(comboost.base_estimator, 'feature_importances_'):
        feature_importances = np.zeros(X_train.shape[1])
        for i, model in enumerate(comboost.models):
            if hasattr(model, 'feature_importances_'):
                feature_importances += model.feature_importances_
        
        feature_importances /= len(comboost.models)
        
        print("\nТоп-10 наиболее важных признаков:")
        top_indices = np.argsort(feature_importances)[-10:][::-1]
        for idx in top_indices:
            print(f"  Признак {idx}: {feature_importances[idx]:.4f}")
        
        # Визуализация важности признаков
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(feature_importances)), feature_importances)
        plt.xlabel('Номер признака')
        plt.ylabel('Важность')
        plt.title('Средняя важность признаков в комитете')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # Анализ согласованности моделей
    predictions_matrix = np.zeros((len(X_test), len(comboost.models)))
    for i, model in enumerate(comboost.models):
        predictions_matrix[:, i] = model.predict(X_test)
    
    # Проверяем, согласны ли все модели в предсказаниях
    agreement_rate = np.mean(np.std(predictions_matrix, axis=1) == 0)
    print(f"\nПроцент примеров, где все модели согласны: {agreement_rate:.2%}")
    
    # Анализ ошибок
    errors = y_test != y_test_pred
    if np.any(errors):
        print(f"\nКоличество ошибок на тесте: {np.sum(errors)}/{len(y_test)} ({np.mean(errors):.2%})")
        
        # Анализируем, какие примеры ошибочны
        error_indices = np.where(errors)[0]
        print(f"Индексы ошибочных примеров (первые 10): {error_indices[:10]}")
        
        # Смотрим уверенность предсказаний для ошибочных примеров
        if hasattr(comboost, 'predict_proba'):
            probas = comboost.predict_proba(X_test[error_indices[:5]])
            print("\nВероятности для первых 5 ошибочных примеров:")
            for i, idx in enumerate(error_indices[:5]):
                true_label = y_test[idx]
                pred_label = y_test_pred[idx]
                print(f"  Пример {idx}: истинный класс={true_label}, предсказанный={pred_label}")
                print(f"    Вероятности: класс -1={probas[i, 0]:.3f}, класс 1={probas[i, 1]:.3f}")
    
    return comboost, X_train, X_test, y_train, y_test

def compare_with_simple_committee():
    """Сравнение ComBoost с вашим исходным простым комитетом"""
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ С ВАШИМ ИСХОДНЫМ КОМИТЕТОМ")
    print("=" * 60)
    
    # Загружаем данные
    X_train, X_test, y_train, y_test = prepare_data_from_your_dataset()
    
    # Ваш исходный подход (простое голосование)
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    
    # Создаем несколько моделей как в вашем коде
    models = [
        DecisionTreeClassifier(max_depth=5, random_state=42),
        RandomForestClassifier(n_estimators=50, random_state=42),
        LogisticRegression(random_state=42),
        DecisionTreeClassifier(max_depth=3, random_state=42),
    ]
    
    # Обучаем каждую модель отдельно
    trained_models = []
    for i, model in enumerate(models):
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        print(f"Модель {i+1} ({type(model).__name__}):")
        print(f"  Точность на обучении: {train_acc:.4f}")
        print(f"  Точность на тесте: {test_acc:.4f}")
        trained_models.append(model)
    
    # Простое голосование (как в вашем исходном коде)
    def simple_committee_predict(models, X):
        predictions = np.zeros((len(X), len(models)))
        for i, model in enumerate(models):
            predictions[:, i] = model.predict(X)
        
        final_predictions = []
        for i in range(len(X)):
            votes = predictions[i, :]
            unique, counts = np.unique(votes, return_counts=True)
            final_predictions.append(unique[np.argmax(counts)])
        
        return np.array(final_predictions)
    
    # Предсказания простого комитета
    y_train_simple = simple_committee_predict(trained_models, X_train)
    y_test_simple = simple_committee_predict(trained_models, X_test)
    
    train_acc_simple = accuracy_score(y_train, y_train_simple)
    test_acc_simple = accuracy_score(y_test, y_test_simple)
    
    print(f"\nПростой комитет (голосование {len(trained_models)} моделей):")
    print(f"  Точность на обучении: {train_acc_simple:.4f}")
    print(f"  Точность на тесте: {test_acc_simple:.4f}")
    
    # ComBoost для сравнения
    comboost = ComBoost(
        base_estimator=DecisionTreeClassifier(max_depth=4, random_state=42),
        T=8,
        verbose=False
    )
    comboost.fit(X_train, y_train)
    
    y_train_comboost = comboost.predict(X_train)
    y_test_comboost = comboost.predict(X_test)
    
    train_acc_comboost = accuracy_score(y_train, y_train_comboost)
    test_acc_comboost = accuracy_score(y_test, y_test_comboost)
    
    print(f"\nComBoost ({len(comboost.models)} моделей):")
    print(f"  Точность на обучении: {train_acc_comboost:.4f}")
    print(f"  Точность на тесте: {test_acc_comboost:.4f}")
    
    # Визуализация сравнения
    models_names = ['ComBoost', 'Простой комитет'] + [f'Модель {i+1}' for i in range(len(trained_models))]
    train_scores = [train_acc_comboost, train_acc_simple] + [accuracy_score(y_train, m.predict(X_train)) for m in trained_models]
    test_scores = [test_acc_comboost, test_acc_simple] + [accuracy_score(y_test, m.predict(X_test)) for m in trained_models]
    
    x = np.arange(len(models_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, train_scores, width, label='Обучающая выборка', color='skyblue')
    rects2 = ax.bar(x + width/2, test_scores, width, label='Тестовая выборка', color='lightcoral')
    
    ax.set_xlabel('Модели')
    ax.set_ylabel('Точность')
    ax.set_title('Сравнение ComBoost с другими подходами')
    ax.set_xticks(x)
    ax.set_xticklabels(models_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Добавляем значения на столбцы
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# ГЛАВНАЯ ФУНКЦИЯ ДЛЯ ЗАПУСКА
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("РЕАЛИЗАЦИЯ ALGORITHM COMBOOST ДЛЯ ВАШЕГО ДАТАСЕТА")
    print("=" * 60)
    
    # Вариант 1: Основной запуск ComBoost на ваших данных
    comboost_model, X_train, X_test, y_train, y_test = run_comboost_on_your_data()
    
    # Вариант 2: Сравнение с вашим исходным подходом (раскомментируйте при необходимости)
    # compare_with_simple_committee()
    
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 60)
    
    # Пример использования обученной модели для новых предсказаний
    print("\nПример использования обученной модели:")
    print(f"Для предсказания на новых данных используйте:")
    print(f"  predictions = comboost_model.predict(новые_данные)")
    print(f"Или для вероятностей:")
    print(f"  probabilities = comboost_model.predict_proba(новые_данные)")