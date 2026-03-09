#!/usr/bin/env python3
"""
Plot federated learning results from comprehensive_results.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

def create_fl_plots(csv_path, output_dir):
    """Create simplified federated learning plots - only performance and precision/recall"""
    
    # Read the data
    df = pd.read_csv(csv_path)
    
    # Clean the data more thoroughly
    df = df[df['mAP50'] > 0].copy()
    
    # Replace any NaN values with 0
    numeric_columns = ['mAP50', 'mAP50-95', 'precision', 'recall']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    print(f"Loaded {len(df)} valid rounds of data")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with only 2 subplots
    fig = plt.figure(figsize=(14, 6))
    
    # 1. Main Performance Metrics Over Rounds
    ax1 = plt.subplot(1, 2, 1)
    plt.plot(df['round_num'], df['mAP50'], marker='o', linewidth=2, markersize=6, label='mAP@50')
    plt.plot(df['round_num'], df['mAP50-95'], marker='s', linewidth=2, markersize=6, label='mAP@50-95')
    plt.xlabel('Federated Round')
    plt.ylabel('Mean Average Precision')
    plt.title('Federated Learning Performance Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Precision vs Recall Trade-off
    ax2 = plt.subplot(1, 2, 2)
    plt.plot(df['round_num'], df['precision'], marker='o', linewidth=2, markersize=6, label='Precision')
    plt.plot(df['round_num'], df['recall'], marker='s', linewidth=2, markersize=6, label='Recall')
    plt.xlabel('Federated Round')
    plt.ylabel('Score')
    plt.title('Precision vs Recall Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the simplified plot
    output_path = Path(output_dir)
    simplified_plot_path = output_path / 'federated_learning_performance.png'
    plt.savefig(simplified_plot_path, dpi=300, bbox_inches='tight')
    print(f"Performance plot saved to: {simplified_plot_path}")
    
    plt.show()

def create_detailed_plots(df, output_dir):
    """Create additional detailed plots"""
    
    # 1. High-resolution performance plot
    plt.figure(figsize=(12, 8))
    plt.plot(df['round_num'], df['mAP50'], marker='o', linewidth=3, markersize=8, label='mAP@50')
    plt.plot(df['round_num'], df['mAP50-95'], marker='s', linewidth=3, markersize=8, label='mAP@50-95')
    
    plt.xlabel('Federated Learning Round', fontsize=14)
    plt.ylabel('Mean Average Precision', fontsize=14)
    plt.title('Federated YOLO Performance Progression', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Annotate best performance
    best_round = df.loc[df['mAP50'].idxmax()]
    plt.annotate(f'Best: Round {int(best_round["round_num"])}\nmAP@50: {best_round["mAP50"]:.3f}',
                xy=(best_round['round_num'], best_round['mAP50']),
                xytext=(best_round['round_num'] + 1, best_round['mAP50'] + 0.05),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    performance_plot_path = output_dir / 'performance_progression_detailed.png'
    plt.savefig(performance_plot_path, dpi=300, bbox_inches='tight')
    print(f"Detailed performance plot saved to: {performance_plot_path}")
    plt.close()
    
    # 2. Precision-Recall scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df['recall'], df['precision'], c=df['round_num'], 
                         cmap='viridis', s=100, alpha=0.7)
    plt.colorbar(scatter, label='Federated Round')
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision vs Recall Evolution\n(Color indicates federated round)', fontsize=14)
    
    # Add trajectory arrows
    for i in range(len(df)-1):
        plt.annotate('', xy=(df.iloc[i+1]['recall'], df.iloc[i+1]['precision']),
                    xytext=(df.iloc[i]['recall'], df.iloc[i]['precision']),
                    arrowprops=dict(arrowstyle='->', alpha=0.5, color='gray'))
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    pr_plot_path = output_dir / 'precision_recall_evolution.png'
    plt.savefig(pr_plot_path, dpi=300, bbox_inches='tight')
    print(f"Precision-Recall plot saved to: {pr_plot_path}")
    plt.close()

def generate_summary_stats(csv_path, output_dir):
    """Generate summary statistics"""
    
    df = pd.read_csv(csv_path)
    df = df[df['mAP50'] > 0].copy()
    
    stats = {
        'total_rounds': len(df),
        'initial_mAP50': df['mAP50'].iloc[0],
        'final_mAP50': df['mAP50'].iloc[-1],
        'best_mAP50': df['mAP50'].max(),
        'best_round': int(df.loc[df['mAP50'].idxmax(), 'round_num']),
        'total_improvement': df['mAP50'].iloc[-1] - df['mAP50'].iloc[0],
        'average_improvement_per_round': (df['mAP50'].iloc[-1] - df['mAP50'].iloc[0]) / len(df)
    }
    
    # Save summary
    summary_path = Path(output_dir) / 'federated_learning_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("FEDERATED LEARNING PERFORMANCE SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Federated Rounds: {stats['total_rounds']}\n")
        f.write(f"Initial mAP@50: {stats['initial_mAP50']:.3f}\n")
        f.write(f"Final mAP@50: {stats['final_mAP50']:.3f}\n")
        f.write(f"Best mAP@50: {stats['best_mAP50']:.3f} (Round {stats['best_round']})\n")
        f.write(f"Total Improvement: +{stats['total_improvement']:.3f}\n")
        f.write(f"Average Improvement per Round: +{stats['average_improvement_per_round']:.4f}\n")
        f.write(f"\nRelative Improvement: {(stats['total_improvement']/stats['initial_mAP50']*100):.1f}%\n")
    
    print(f"Summary statistics saved to: {summary_path}")
    return stats

def main():
    # Configuration
    csv_file = "output_yolo_temp/global/evaluation_metrics/comprehensive_results.csv"
    output_directory = "output_yolo_temp/global/evaluation_metrics/"
    
    # Check if CSV exists
    if not Path(csv_file).exists():
        print(f"Error: CSV file not found at {csv_file}")
        return
    
    print("Creating federated learning analysis plots...")
    
    # Create plots
    create_fl_plots(csv_file, output_directory)
    
    # Generate summary statistics
    stats = generate_summary_stats(csv_file, output_directory)
    
    print("\nFederated Learning Analysis Complete!")
    print(f"Initial mAP@50: {stats['initial_mAP50']:.3f}")
    print(f"Final mAP@50: {stats['final_mAP50']:.3f}")
    print(f"Total Improvement: +{stats['total_improvement']:.3f}")

if __name__ == "__main__":
    main()