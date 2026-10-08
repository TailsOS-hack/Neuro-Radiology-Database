"""Compare fixed-cohort predictions while preserving historical outputs by default."""

import argparse
import glob
import os
from pathlib import Path

# Paths
GT_PATH = "data/evaluation/ground_truth/radiologist_test_key.csv"
RAD_DIR = "data/evaluation/radiologist_results/"
MODEL_PATH = "data/evaluation/model_results/model_predictions.csv"
OUTPUT_DIR = "data_visualization/comparison"


# Standard Classes
classes = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
class_names_short = ['Normal', 'Very Mild', 'Mild', 'Moderate']

# Map Rad Labels
rad_map = {
    'NonDemented': 'NonDemented',
    'VeryMild': 'VeryMildDemented',
    'Mild': 'MildDemented',
    'Moderate': 'ModerateDemented',
    'Non Demented': 'NonDemented',
    'Very Mild': 'VeryMildDemented',
    'Moderate Demented': 'ModerateDemented'
}

def load_clean_sheet(filepath, sheet_name):
    """Loads a specific sheet and returns a cleaned DataFrame."""
    import pandas as pd

    try:
        df = pd.read_excel(filepath, sheet_name=sheet_name)
    except Exception as e:
        print(f"Error reading {filepath} sheet {sheet_name}: {e}")
        return None

    # Find diagnosis column
    rad_cols = [c for c in df.columns if "Diagnosis" in c]
    if not rad_cols:
        return None

    rad_col = rad_cols[0]

    # Check if empty
    if df[rad_col].count() < 50: # Threshold to skip empty/template sheets
        return None

    df = df[['FileName', rad_col]].copy()
    df.rename(columns={rad_col: 'RadPrediction'}, inplace=True)
    df['FileName'] = df['FileName'].astype(str).str.strip()

    # Map predictions
    df['RadPrediction'] = df['RadPrediction'].astype(str).str.strip()
    df['RadPrediction_Mapped'] = df['RadPrediction'].map(rad_map)

    # Fallback for exact matches
    mask = df['RadPrediction_Mapped'].isna() & df['RadPrediction'].isin(classes)
    df.loc[mask, 'RadPrediction_Mapped'] = df.loc[mask, 'RadPrediction']

    return df

def load_comparison_data(ground_truth_path, model_predictions_path):
    """Require exactly one valid prediction for every ground-truth filename."""
    import pandas as pd

    ground_truth = pd.read_csv(ground_truth_path, dtype=str, keep_default_na=False)
    predictions = pd.read_csv(model_predictions_path, dtype=str, keep_default_na=False)
    for source, frame, label_column in [
        ("Ground truth", ground_truth, "TrueCategory"),
        ("Model predictions", predictions, "ModelPrediction"),
    ]:
        required = {"FileName", label_column}
        missing_columns = required - set(frame.columns)
        if missing_columns:
            raise ValueError(f"{source} is missing columns: {sorted(missing_columns)}")
        if frame.empty:
            raise ValueError(f"{source} has no rows")
        for column in required:
            frame[column] = frame[column].str.strip()
        if frame["FileName"].eq("").any():
            raise ValueError(f"{source} has empty filenames")
        duplicates = frame.loc[frame["FileName"].duplicated(), "FileName"].tolist()
        if duplicates:
            raise ValueError(f"{source} has duplicate filenames: {duplicates}")
        unknown = set(frame[label_column]) - set(classes)
        if unknown:
            raise ValueError(f"{source} has unknown dementia labels: {sorted(unknown)}")

    expected = set(ground_truth["FileName"])
    actual = set(predictions["FileName"])
    if expected != actual:
        raise ValueError(
            "Prediction filename coverage does not match ground truth; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    # The ground-truth file is authoritative; prediction exports may also carry
    # a TrueCategory field, which must not shadow or duplicate the actual key.
    return ground_truth[["FileName", "TrueCategory"]].merge(
        predictions[["FileName", "ModelPrediction"]],
        on="FileName", how="inner", validate="one_to_one",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-predictions", default=MODEL_PATH,
                        help="Prediction CSV; the historical CSV remains the default")
    parser.add_argument("--output-dir", default=None,
                        help="Figure directory; custom predictions default to comparison_current")
    args = parser.parse_args(argv)
    custom_model = Path(args.model_predictions).resolve() != Path(MODEL_PATH).resolve()
    output_dir = args.output_dir or (OUTPUT_DIR + "_current" if custom_model else OUTPUT_DIR)
    print("Loading GT and Model...")
    main_df = load_comparison_data(GT_PATH, args.model_predictions)
    benchmark_note = ""
    if custom_model:
        benchmark_note = (
            f"Reused {len(main_df)}-image benchmark; includes training/validation images."
            "\nThis is not an independent clinical evaluation."
        )
        print(benchmark_note)

    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)

    def save_chart(filename):
        if benchmark_note:
            plt.gcf().text(0.5, 0.012, benchmark_note, ha="center", va="bottom", fontsize=9)
            plt.tight_layout(rect=(0, 0.075, 1, 1))
        else:
            plt.tight_layout()
        plt.savefig(Path(output_dir) / filename, dpi=300)
        plt.close()

    model_acc = accuracy_score(main_df['TrueCategory'], main_df['ModelPrediction'])
    print(f"{'Reused benchmark model accuracy' if custom_model else 'AI Model Accuracy'}: {model_acc:.4f}")

    # 2. Load All Radiologists
    rad_files = glob.glob(os.path.join(RAD_DIR, "*.xlsx"))
    rad_results = []

    processed_ids = set()

    print(f"Scanning {len(rad_files)} radiologist files...")

    for fpath in rad_files:
        fname = os.path.basename(fpath)
        try:
            xls = pd.ExcelFile(fpath)
            for sheet in xls.sheet_names:
                r_df = load_clean_sheet(fpath, sheet)
                if r_df is not None:
                    rad_id = sheet # Use sheet name as ID (Kalapos, Thamburaj, Kanekar)

                    # Deduplicate if same ID found in multiple files
                    if rad_id in processed_ids:
                        print(f"Skipping duplicate Rad ID: {rad_id} in {fname}")
                        continue

                    # Merge with main_df to ensure alignment
                    temp_df = main_df.merge(r_df, on="FileName", how="left", validate="one_to_one")
                    temp_df['RadPrediction_Mapped'] = temp_df['RadPrediction_Mapped'].fillna("Unknown")

                    acc = accuracy_score(temp_df['TrueCategory'], temp_df['RadPrediction_Mapped'])
                    print(f"Loaded {rad_id} from {fname}: Accuracy = {acc:.4f}")

                    rad_results.append({
                        'id': rad_id,
                        'accuracy': acc,
                        'df': r_df
                    })
                    processed_ids.add(rad_id)

        except Exception as e:
            print(f"Error processing file {fname}: {e}")

    # 3. Aggregation Logic
    if not rad_results:
        print("No valid radiologist results found. Exiting.")
        return

    # Sort by accuracy descending
    rad_results.sort(key=lambda x: x['accuracy'], reverse=True)

    best_rad = rad_results[0]
    worst_rad = rad_results[-1]

    # Calculate Average (Mean) Accuracy
    all_accs = [r['accuracy'] for r in rad_results]
    mean_all = np.mean(all_accs)
    print(f"Average Radiologist Accuracy: {mean_all:.4f}")

    # --- VISUALIZATION ---
    print("Generating visualizations...")
    sns.set_style("whitegrid")

    # 1. Comprehensive Chart
    viz_data = []
    for r in rad_results:
        viz_data.append({'Participant': r['id'], 'Accuracy': r['accuracy'], 'Type': 'Individual'})
    viz_data.append({'Participant': 'All Rads (Avg)', 'Accuracy': mean_all, 'Type': 'Aggregate'})
    viz_data.append({'Participant': 'AI Model', 'Accuracy': model_acc, 'Type': 'AI'})
    viz_df = pd.DataFrame(viz_data)

    plt.figure(figsize=(12, 7))
    palette = {}
    for i, row in viz_df.iterrows():
        if row['Type'] == 'Individual': palette[row['Participant']] = '#3498db'
        elif row['Type'] == 'Aggregate': palette[row['Participant']] = '#2ecc71'
        elif row['Type'] == 'AI': palette[row['Participant']] = '#e74c3c'

    ax = sns.barplot(x='Participant', y='Accuracy', data=viz_df, palette=palette)
    plt.title('Reused Benchmark Accuracy: Radiologists and AI' if custom_model else 'Diagnostic Accuracy Comparison: Radiologists vs AI (Comprehensive)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1.1)
    plt.xticks(rotation=15)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1%}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=12, fontweight='bold', xytext=(0, 5), textcoords='offset points')
    save_chart("accuracy_comparison_comprehensive.png")

    # 2. Average Radiologists vs AI
    avg_data = [
        {'Participant': 'Average Radiologist', 'Accuracy': mean_all},
        {'Participant': 'AI Model', 'Accuracy': model_acc}
    ]
    avg_df = pd.DataFrame(avg_data)

    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x='Participant', y='Accuracy', data=avg_df, palette={'Average Radiologist': '#2ecc71', 'AI Model': '#e74c3c'})
    plt.title('Average Radiologist vs AI Model', fontsize=16, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1.1)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1%}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=12, fontweight='bold', xytext=(0, 5), textcoords='offset points')
    save_chart("accuracy_avg_vs_ai.png")

    # 3. Best Rad vs Worst Rad vs AI
    range_data = [
        {'Participant': f"Best Rad ({best_rad['id']})", 'Accuracy': best_rad['accuracy']},
        {'Participant': f"Worst Rad ({worst_rad['id']})", 'Accuracy': worst_rad['accuracy']},
        {'Participant': 'AI Model', 'Accuracy': model_acc}
    ]
    range_df = pd.DataFrame(range_data)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Participant', y='Accuracy', data=range_df, palette={
        f"Best Rad ({best_rad['id']})": '#3498db',
        f"Worst Rad ({worst_rad['id']})": '#95a5a6',
        'AI Model': '#e74c3c'
    })
    plt.title('Performance Range: Best & Worst Radiologist vs AI', fontsize=16, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1.1)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1%}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=12, fontweight='bold', xytext=(0, 5), textcoords='offset points')
    save_chart("accuracy_range_comparison.png")

    # 4. Per-Class Sensitivity (All 4 Participants)
    recall_ai = recall_score(main_df['TrueCategory'], main_df['ModelPrediction'], labels=classes, average=None)

    per_class_data = []
    for i, cls in enumerate(class_names_short):
        per_class_data.append({'Class': cls, 'Recall': recall_ai[i], 'Participant': 'AI Model'})

    for r in rad_results:
        temp_df = main_df.merge(r['df'], on="FileName", how="left", validate="one_to_one")
        temp_df['RadPrediction_Mapped'] = temp_df['RadPrediction_Mapped'].fillna("Unknown")
        rec = recall_score(temp_df['TrueCategory'], temp_df['RadPrediction_Mapped'], labels=classes, average=None)
        for i, cls in enumerate(class_names_short):
            per_class_data.append({'Class': cls, 'Recall': rec[i], 'Participant': r['id']})

    pc_df = pd.DataFrame(per_class_data)

    plt.figure(figsize=(14, 7))
    palette_pc = {'AI Model': '#e74c3c'}
    rad_colors = ['#3498db', '#2ecc71', '#f1c40f']
    for i, r in enumerate(rad_results):
        palette_pc[r['id']] = rad_colors[i % len(rad_colors)]

    ax = sns.barplot(x='Class', y='Recall', hue='Participant', data=pc_df, palette=palette_pc)
    plt.title('Per-Class Sensitivity (Recall): Detailed Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Sensitivity', fontsize=12)
    plt.ylim(0, 1.15)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)

    save_chart("per_class_sensitivity_detailed.png")

    # 5. Confusion Matrices
    num_rads = len(rad_results)
    total_plots = num_rads + 1
    cols = 2
    rows = (total_plots + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(16, 7 * rows))
    axes = axes.flatten()

    for i, r in enumerate(rad_results):
        temp_df = main_df.merge(r['df'], on="FileName", how="left", validate="one_to_one")
        temp_df['RadPrediction_Mapped'] = temp_df['RadPrediction_Mapped'].fillna("Unknown")
        cm = confusion_matrix(temp_df['TrueCategory'], temp_df['RadPrediction_Mapped'], labels=classes)

        # Normalize to percentages (row-wise)
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', xticklabels=class_names_short, yticklabels=class_names_short, ax=axes[i], cbar=False)
        axes[i].set_title(f"Neuroradiologist {i+1}", fontsize=14, fontweight='bold')

    ai_idx = num_rads
    cm_model = confusion_matrix(main_df['TrueCategory'], main_df['ModelPrediction'], labels=classes)

    # Normalize AI matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_model_norm = cm_model.astype('float') / cm_model.sum(axis=1)[:, np.newaxis]
    cm_model_norm = np.nan_to_num(cm_model_norm)

    # (Manual override removed)

    sns.heatmap(cm_model_norm, annot=True, fmt='.2%', cmap='Reds', xticklabels=class_names_short, yticklabels=class_names_short, ax=axes[ai_idx], cbar=False)
    axes[ai_idx].set_title(f"AI Model", fontsize=14, fontweight='bold')

    for j in range(ai_idx + 1, len(axes)):
        axes[j].axis('off')

    save_chart("confusion_matrix_comparison.png")

    print("All charts generated successfully.")


if __name__ == "__main__":
    main()
