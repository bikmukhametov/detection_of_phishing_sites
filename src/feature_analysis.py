"""
Analysis and visualization of feature selection results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional


def save_feature_selection_results(results: Dict, output_dir: Path) -> None:
    """
    Save consolidated feature selection results.

    Policy:
    - Produce a single markdown report (`feature_selection_summary.md`).
    - Do NOT create duplicate `.csv`, `.json` or per-algorithm `*_features.txt` files.
    - Leave plot generation to the plotting functions (they create PNGs).

    Args:
        results: Dictionary with results from all algorithms
        output_dir: Output directory path
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect available feature names from results (fallback if full list not provided)
    all_names = set()
    for v in results.values():
        names = v.get('selected_feature_names')
        if names:
            all_names.update(names)
    all_feature_names = sorted(all_names)

    # Prefer using the detailed markdown creator if available in this module
    try:
        create_summary_markdown(results, output_dir, all_feature_names)
        return
    except Exception:
        # Fallback: write a minimal markdown summary
        out_file = output_dir / 'feature_selection_summary.md'
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write('# Feature selection summary\n\n')
            f.write('This report consolidates selected features and key metrics for each algorithm.\n\n')
            f.write('## Algorithms\n\n')
            for algo, r in results.items():
                f.write(f'### {algo}\n')
                sel = r.get('selected_feature_names', [])
                f.write(f'- Selected features ({len(sel)}):\n')
                for s in sel:
                    f.write(f'  - {s}\n')
                m = r.get('metrics', {})
                f.write(f'- Metrics: accuracy={m.get("accuracy", 0):.4f}, precision={m.get("precision", 0):.4f}, recall={m.get("recall", 0):.4f}, f1={m.get("f1", 0):.4f}, roc_auc={m.get("roc_auc", 0):.4f}\n\n')
        return


def plot_algorithm_comparison(results: Dict, output_dir: Path) -> None:
    """
    Create comparison plots for all algorithms.
    
    Args:
        results: Dictionary with results from all algorithms
        output_dir: Output directory path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    algo_names = list(results.keys())
    
    # Prepare data
    metrics_data = {
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1 Score': [],
        'ROC-AUC': [],
    }
    feature_counts = []
    
    for algo_name in algo_names:
        metrics = results[algo_name]['metrics']
        metrics_data['Accuracy'].append(metrics['accuracy'])
        metrics_data['Precision'].append(metrics['precision'])
        metrics_data['Recall'].append(metrics['recall'])
        metrics_data['F1 Score'].append(metrics['f1'])
        metrics_data['ROC-AUC'].append(metrics['roc_auc'])
        feature_counts.append(len(results[algo_name]['selected_features']))
    
    # Plot 1: Metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Сравнение метрик качества алгоритмов отбора признаков', 
                 fontsize=14, fontweight='bold')
    
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    for idx, (ax, metric) in enumerate(zip(axes.flat, metrics_to_plot)):
        bars = ax.bar(algo_names, metrics_data[metric], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f'{metric} по алгоритмам', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
        
        ax.set_xticklabels(algo_names, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'algorithms_metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: ROC-AUC and Feature Count comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    bars1 = ax1.bar(algo_names, metrics_data['ROC-AUC'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_ylabel('ROC-AUC', fontsize=11)
    ax1.set_title('ROC-AUC по алгоритмам', fontsize=12, fontweight='bold')
    ax1.set_ylim([0, 1.0])
    ax1.grid(axis='y', alpha=0.3)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)
    ax1.set_xticklabels(algo_names, rotation=45, ha='right')
    
    bars2 = ax2.bar(algo_names, feature_counts, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax2.set_ylabel('Количество выбранных признаков', fontsize=11)
    ax2.set_title('Количество отобранных признаков', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)
    ax2.set_xticklabels(algo_names, rotation=45, ha='right')
    
    fig.suptitle('ROC-AUC и количество признаков', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_auc_and_feature_count.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_quality_vs_feature_count(results: Dict, output_dir: Path) -> None:
    """
    Plot dependency of quality criterion Q on number of features.
    
    Args:
        results: Dictionary with results from all algorithms
        output_dir: Output directory path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, (algo_name, algo_results) in enumerate(results.items()):
        tracker = algo_results['stats']['tracker']
        
        # X-axis: number of selected features
        feature_counts = [len(selected) for selected in tracker.selected_features_history]
        # Y-axis: Q (error rate)
        quality_values = tracker.quality_history
        
        ax.plot(feature_counts, quality_values, 'o-', 
               label=algo_name, linewidth=2.5, markersize=6, 
               color=colors[idx % len(colors)], alpha=0.8)
    
    ax.set_xlabel('Количество выбранных признаков (n)', fontsize=12)
    ax.set_ylabel('Q - критерий качества (ошибка на обучении)', fontsize=12)
    ax.set_title('Зависимость функционала качества Q от количества признаков', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'quality_vs_feature_count.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_convergence_curves(results: Dict, output_dir: Path) -> None:
    """
    Plot convergence curves showing Q over iterations for each algorithm.
    
    Args:
        results: Dictionary with results from all algorithms
        output_dir: Output directory path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, (algo_name, algo_results) in enumerate(results.items()):
        tracker = algo_results['stats']['tracker']
        iterations = tracker.iterations
        quality_values = tracker.quality_history
        
        axes[idx].plot(iterations, quality_values, 'o-', 
                      linewidth=2, markersize=4, color=colors[idx], alpha=0.8)
        axes[idx].set_xlabel('Итерация', fontsize=11)
        axes[idx].set_ylabel('Q - критерий качества', fontsize=11)
        axes[idx].set_title(f'{algo_name}', fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, linestyle='--')
        
        # Add final value annotation
        final_q = quality_values[-1] if quality_values else 0
        axes[idx].text(0.95, 0.95, f'Финальное Q: {final_q:.4f}',
                      transform=axes[idx].transAxes,
                      verticalalignment='top', horizontalalignment='right',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                      fontsize=10)
    
    fig.suptitle('Кривые сходимости алгоритмов', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'convergence_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_count_progression(results: Dict, output_dir: Path) -> None:
    """
    Plot how the number of selected features changes over iterations.
    
    Args:
        results: Dictionary with results from all algorithms
        output_dir: Output directory path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, (algo_name, algo_results) in enumerate(results.items()):
        tracker = algo_results['stats']['tracker']
        iterations = tracker.iterations
        feature_counts = [len(selected) for selected in tracker.selected_features_history]
        
        axes[idx].plot(iterations, feature_counts, 'o-', 
                      linewidth=2, markersize=4, color=colors[idx], alpha=0.8)
        axes[idx].set_xlabel('Итерация', fontsize=11)
        axes[idx].set_ylabel('Количество признаков', fontsize=11)
        axes[idx].set_title(f'{algo_name}', fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, linestyle='--')
        axes[idx].axhline(y=feature_counts[-1] if feature_counts else 0, 
                         color='red', linestyle='--', alpha=0.5)
    
    fig.suptitle('Изменение количества отобранных признаков по итерациям', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_count_progression.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_summary_markdown(results: Dict, output_dir: Path, all_feature_names: List[str]) -> None:
    """
    Create a markdown summary of feature selection results.
    
    Args:
        results: Dictionary with results from all algorithms
        output_dir: Output directory path
        all_feature_names: List of all available features
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'feature_selection_summary.md', 'w', encoding='utf-8') as f:
        f.write('# Результаты отбора признаков\n\n')
        
        # Overall comparison table
        f.write('## Сравнение алгоритмов\n\n')
        f.write('| Алгоритм | Признаков | Accuracy | Precision | Recall | F1 | ROC-AUC |\n')
        f.write('|----------|-----------|----------|-----------|--------|----|---------|\n')
        
        for algo_name in results.keys():
            algo_result = results[algo_name]
            metrics = algo_result['metrics']
            count = len(algo_result['selected_features'])
            f.write(f'| {algo_name} | {count} | {metrics["accuracy"]:.4f} | '
                   f'{metrics["precision"]:.4f} | {metrics["recall"]:.4f} | '
                   f'{metrics["f1"]:.4f} | {metrics["roc_auc"]:.4f} |\n')
        
        f.write('\n')
        
        # Detailed results for each algorithm
        for algo_name in results.keys():
            algo_result = results[algo_name]
            selected_features = algo_result['selected_feature_names']
            metrics = algo_result['metrics']
            stats = algo_result['stats']
            
            f.write(f'## {algo_name}\n\n')
            f.write(f'**Статистика:**\n')
            f.write(f'- Количество отобранных признаков: {len(selected_features)}\n')
            f.write(f'- Финальное значение Q (ошибка): {stats["final_quality"]:.4f}\n')
            f.write(f'- Итераций: {stats.get("iterations", stats.get("generations", "N/A"))}\n\n')
            
            f.write(f'**Метрики качества:**\n')
            f.write(f'- Accuracy: {metrics["accuracy"]:.4f}\n')
            f.write(f'- Precision: {metrics["precision"]:.4f}\n')
            f.write(f'- Recall: {metrics["recall"]:.4f}\n')
            f.write(f'- F1 Score: {metrics["f1"]:.4f}\n')
            f.write(f'- ROC-AUC: {metrics["roc_auc"]:.4f}\n\n')
            
            f.write(f'**Выбранные признаки ({len(selected_features)}):**\n')
            for i, feat in enumerate(selected_features, 1):
                f.write(f'{i}. {feat}\n')
            
            f.write('\n---\n\n')
