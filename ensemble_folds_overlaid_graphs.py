import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import logging
import glob

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

def create_overlay_plots(base_dir):
    """
    Create overlay plots using existing data
    """
    # Set up paths
    save_dir = os.path.join(base_dir, "saved_models")
    test_results_dir = os.path.join(save_dir, "test_evaluation")
    dyn_ensemble_dir = os.path.join(test_results_dir, "dynamic_ensemble_plots")
    overlay_dir = os.path.join(test_results_dir, "overlay_plots")
    
    os.makedirs(overlay_dir, exist_ok=True)
    
    # Load test results
    test_results_path = os.path.join(test_results_dir, "test_results.json")
    with open(test_results_path, 'r') as f:
        test_results = json.load(f)
    
    # Load dynamic ensemble results
    if 'dynamic_ensemble' not in test_results:
        logging.error("Dynamic ensemble results not found in test_results.json")
        return
    
    dynamic_ensemble_results = test_results['dynamic_ensemble']
    
    # Load ensemble probabilities and labels
    ensemble_file = os.path.join(dyn_ensemble_dir, "dynamic_ensemble_probs_labels.npz")
    if not os.path.exists(ensemble_file):
        logging.error(f"Dynamic ensemble predictions file not found: {ensemble_file}")
        return
    
    ensemble_data = np.load(ensemble_file)
    ensemble_probs = ensemble_data['probabilities']
    all_labels = ensemble_data['labels']
    
    # Load fold probabilities
    fold_probs = {}
    for fold_num in range(5):  # Assuming 5 folds
        fold_file = os.path.join(dyn_ensemble_dir, f"fold_{fold_num}_probs.npz")
        if os.path.exists(fold_file):
            data = np.load(fold_file)
            fold_probs[fold_num] = data['probabilities']
            logging.info(f"Loaded fold {fold_num} probabilities")
    
    if not fold_probs:
        # Try loading fold probabilities from fold-specific directories
        for fold_num in range(5):
            fold_dir = os.path.join(test_results_dir, f"fold_{fold_num}")
            if os.path.exists(fold_dir):
                fold_file = os.path.join(fold_dir, "external_test_probs_labels.npz")
                if os.path.exists(fold_file):
                    data = np.load(fold_file)
                    fold_probs[fold_num] = data['probabilities']
                    logging.info(f"Loaded fold {fold_num} probabilities from fold directory")
    
    if not fold_probs:
        logging.error("No fold probabilities found, cannot create overlay plots")
        return
    
    logging.info(f"Creating overlay plots with {len(fold_probs)} folds...")
    
    # Call the function to create the overlay plots
    create_ensemble_vs_folds_overlay_plots(fold_probs, all_labels, dynamic_ensemble_results, test_results, base_dir)

def create_ensemble_vs_folds_overlay_plots(fold_probs, all_labels, dynamic_ensemble_results, test_results, base_dir):
    """
    Create overlay plots comparing all individual folds with the dynamic ensemble
    """
    # Create directory for overlay plots
    overlay_dir = os.path.join(base_dir, "saved_models", "test_evaluation", "overlay_plots")
    os.makedirs(overlay_dir, exist_ok=True)
    
    # Get dynamic ensemble probabilities
    dyn_ensemble_dir = os.path.join(base_dir, "saved_models", "test_evaluation", "dynamic_ensemble_plots")
    ensemble_file = os.path.join(dyn_ensemble_dir, "dynamic_ensemble_probs_labels.npz")
    
    if not os.path.exists(ensemble_file):
        logging.error("Dynamic ensemble predictions file not found for overlay plots")
        return
    
    ensemble_data = np.load(ensemble_file)
    ensemble_probs = ensemble_data['probabilities']
    
    # 1. ROC Curves Overlay
    plt.figure(figsize=(12, 10))
    
    # Colors for different folds
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    line_styles = ['-', '--', '-.', ':', '-', '--']
    
    # Plot ROC curve for each fold
    for i, (fold_num, probs) in enumerate(sorted(fold_probs.items())):
        fpr, tpr, _ = roc_curve(all_labels, probs)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=colors[i % len(colors)], 
                linestyle=line_styles[i % len(line_styles)],
                lw=2, label=f'Fold {fold_num} (AUC = {roc_auc:.3f})')
    
    # Add dynamic ensemble ROC
    fpr, tpr, _ = roc_curve(all_labels, ensemble_probs)
    roc_auc = dynamic_ensemble_results['metrics']['auc']
    plt.plot(fpr, tpr, color='black', linestyle='-', lw=3,
            label=f'Dynamic Ensemble (AUC = {roc_auc:.3f})')
    
    # Add reference line
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    
    # Customize plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curves: Individual Folds vs Dynamic Ensemble', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(overlay_dir, 'roc_curves_folds_vs_ensemble_overlay.png'), dpi=300)
    plt.savefig(os.path.join(overlay_dir, 'roc_curves_folds_vs_ensemble_overlay.pdf'))
    plt.close()
    
    # 2. Precision-Recall Curves Overlay
    plt.figure(figsize=(12, 10))
    
    # Plot PR curve for each fold
    for i, (fold_num, probs) in enumerate(sorted(fold_probs.items())):
        precision, recall, _ = precision_recall_curve(all_labels, probs)
        pr_auc = np.trapz(precision, recall)
        
        plt.plot(recall, precision, color=colors[i % len(colors)], 
                linestyle=line_styles[i % len(line_styles)],
                lw=2, label=f'Fold {fold_num} (AP = {pr_auc:.3f})')
    
    # Add dynamic ensemble PR curve
    precision, recall, _ = precision_recall_curve(all_labels, ensemble_probs)
    pr_auc = dynamic_ensemble_results['metrics']['avg_precision']
    plt.plot(recall, precision, color='black', linestyle='-', lw=3,
            label=f'Dynamic Ensemble (AP = {pr_auc:.3f})')
    
    # Customize plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curves: Individual Folds vs Dynamic Ensemble', fontsize=16)
    plt.legend(loc="best", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(overlay_dir, 'pr_curves_folds_vs_ensemble_overlay.png'), dpi=300)
    plt.savefig(os.path.join(overlay_dir, 'pr_curves_folds_vs_ensemble_overlay.pdf'))
    plt.close()
    
    # 3. Create metrics comparison bar chart
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'specificity', 'mcc']
    fold_results = {k: v for k, v in test_results.items() if k.startswith('fold_')}
    
    for metric in metrics:
        plt.figure(figsize=(14, 8))
        
        # Extract values for each fold
        fold_nums = []
        fold_values = []
        
        for fold_key, fold_data in sorted(fold_results.items()):
            fold_num = int(fold_key.split('_')[1])
            fold_nums.append(fold_num)
            value = fold_data['metrics'].get(metric, 0)
            fold_values.append(value)
        
        # Add ensemble value
        ensemble_value = dynamic_ensemble_results['metrics'].get(metric, 0)
        
        # Create bar positions
        x = np.arange(len(fold_nums) + 1)  # Folds + ensemble
        
        # Plot bars
        plt.bar(x[:-1], fold_values, color='steelblue', alpha=0.7, label='Individual Folds')
        plt.bar(x[-1], [ensemble_value], color='darkred', alpha=0.7, label='Dynamic Ensemble')
        
        # Add value labels
        for i, value in enumerate(fold_values + [ensemble_value]):
            plt.text(i, value + 0.01, f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Add fold average line
        avg_fold_value = np.mean(fold_values)
        plt.axhline(y=avg_fold_value, color='navy', linestyle='--', 
                   label=f'Fold Average: {avg_fold_value:.3f}')
        
        # Customize plot
        plt.xlabel('Model', fontsize=14)
        plt.ylabel(f'{metric.capitalize()} Score', fontsize=14)
        plt.title(f'{metric.capitalize()}: Individual Folds vs Dynamic Ensemble', fontsize=16)
        plt.xticks(x, [f'Fold {num}' for num in fold_nums] + ['Ensemble'])
        plt.legend(loc='best')
        plt.grid(axis='y', alpha=0.3)
        
        # Set ylim based on metric
        if metric == 'mcc':
            plt.ylim(-1.1, 1.1)
        else:
            plt.ylim(0, 1.1)
        
        plt.savefig(os.path.join(overlay_dir, f'{metric}_folds_vs_ensemble_comparison.png'), dpi=300)
        plt.savefig(os.path.join(overlay_dir, f'{metric}_folds_vs_ensemble_comparison.pdf'))
        plt.close()
    
    # 4. Radar chart comparing all folds and ensemble
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'specificity']
    metrics_nice_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'Specificity']
    
    # Set up the radar chart
    n_metrics = len(metrics)
    angles = np.linspace(0, 2*np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
    
    # Plot data for each fold
    for i, (fold_key, fold_data) in enumerate(sorted(fold_results.items())):
        fold_num = int(fold_key.split('_')[1])
        
        # Get metrics for this fold
        fold_values = []
        for m in metrics:
            if m in fold_data['metrics']:
                fold_values.append(fold_data['metrics'][m])
            else:
                fold_values.append(0)  # Default if metric not found
                
        fold_values += fold_values[:1]  # Close the polygon
        
        # Plot fold data
        ax.plot(angles, fold_values, color=colors[i % len(colors)], 
                linestyle=line_styles[i % len(line_styles)], linewidth=2, 
                label=f'Fold {fold_num}')
        ax.fill(angles, fold_values, color=colors[i % len(colors)], alpha=0.1)
    
    # Add ensemble data
    ensemble_values = []
    for m in metrics:
        if m in dynamic_ensemble_results['metrics']:
            ensemble_values.append(dynamic_ensemble_results['metrics'][m])
        else:
            ensemble_values.append(0)
            
    ensemble_values += ensemble_values[:1]  # Close the polygon
    
    ax.plot(angles, ensemble_values, color='darkred', linewidth=3, label='Dynamic Ensemble')
    ax.fill(angles, ensemble_values, color='darkred', alpha=0.1)
    
    # Set up the radar chart properties
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_nice_names, fontsize=12)
    
    # Add y-axis grid lines at 0.2 intervals
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_ylim(0, 1)
    
    # Add a title
    plt.title('Performance Metrics: Individual Folds vs Dynamic Ensemble', fontsize=16, y=1.08)
    
    # Add a legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.savefig(os.path.join(overlay_dir, 'metrics_radar_folds_vs_ensemble.png'), dpi=300)
    plt.savefig(os.path.join(overlay_dir, 'metrics_radar_folds_vs_ensemble.pdf'))
    plt.close()
    
    # 5. Create statistical comparison report
    stats_path = os.path.join(overlay_dir, "folds_vs_ensemble_statistical_comparison.txt")
    with open(stats_path, 'w') as f:
        f.write("STATISTICAL COMPARISON: INDIVIDUAL FOLDS VS DYNAMIC ENSEMBLE\n\n")
        
        for metric in metrics:
            f.write(f"===== {metric.upper()} =====\n")
            
            # Get fold values
            fold_values = [fold_data['metrics'].get(metric, 0) for _, fold_data in sorted(fold_results.items())]
            ensemble_value = dynamic_ensemble_results['metrics'].get(metric, 0)
            
            # Calculate statistics
            fold_avg = np.mean(fold_values)
            fold_std = np.std(fold_values)
            fold_min = np.min(fold_values)
            fold_max = np.max(fold_values)
            
            # Statistical improvement over average
            improvement_over_avg = ensemble_value - fold_avg
            relative_improvement = (improvement_over_avg / fold_avg) * 100 if fold_avg > 0 else 0
            
            # Count how many folds ensemble beats
            better_than_count = sum(1 for val in fold_values if ensemble_value > val)
            total_folds = len(fold_values)
            
            # Write statistics
            f.write(f"Fold average: {fold_avg:.4f} ± {fold_std:.4f}\n")
            f.write(f"Fold range: [{fold_min:.4f}, {fold_max:.4f}]\n")
            f.write(f"Dynamic Ensemble: {ensemble_value:.4f}\n")
            f.write(f"Improvement over average: {improvement_over_avg:.4f} ({relative_improvement:.2f}%)\n")
            f.write(f"Ensemble beats {better_than_count}/{total_folds} individual folds\n\n")
        
        # Overall conclusion
        f.write("OVERALL CONCLUSION:\n")
        f.write("The dynamic weighted ensemble combines predictions from all folds using learned weights.\n")
        
        # Count metrics where ensemble beats the average
        metrics_better_than_avg = 0
        for metric in metrics:
            fold_values = [fold_data['metrics'].get(metric, 0) for _, fold_data in sorted(fold_results.items())]
            ensemble_value = dynamic_ensemble_results['metrics'].get(metric, 0)
            fold_avg = np.mean(fold_values)
            
            if ensemble_value > fold_avg:
                metrics_better_than_avg += 1
        
        f.write(f"The ensemble outperforms the fold average in {metrics_better_than_avg}/{len(metrics)} metrics.\n")
    
    logging.info(f"Created fold vs ensemble overlay plots and statistical comparisons in {overlay_dir}")
    return

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Create overlay plots for skin lesion classification.')
    parser.add_argument('--base_dir', type=str, default="/scratch365/mli29/combined_bayesian",
                        help='Base directory for the experiment')
    args = parser.parse_args()
    
    create_overlay_plots(args.base_dir)