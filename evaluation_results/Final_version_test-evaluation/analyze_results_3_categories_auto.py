import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, fbeta_score, cohen_kappa_score
import os
import io
import argparse

def calculate_precision_recall(conf_matrix, labels):
    """
    Calculates precision and recall for each class from a confusion matrix.

    Note:
    - Recall is also known as "Sensitivity" or the True Positive Rate.
    - Precision is the measure of a model's accuracy in classifying a sample as positive.

    Args:
        conf_matrix (np.ndarray): The confusion matrix.
        labels (list): The list of class labels, corresponding to the matrix indices.

    Returns:
        pd.DataFrame: A DataFrame with precision and recall for each class.
    """
    metrics = []
    for i, label in enumerate(labels):
        TP = conf_matrix[i, i]
        FP = conf_matrix[:, i].sum() - TP
        FN = conf_matrix[i, :].sum() - TP
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        metrics.append({
            'sub_category': label,
            'precision': precision,
            'recall': recall,
            'TP': TP,
            'FP': FP,
            'FN': FN
        })
    
    return pd.DataFrame(metrics)

def analyze_predictions(result_filepath, ground_truth_filepath, output_filepath,
                        beta_decision, beta_lane, beta_obstacle,
                        weight_decision, weight_lane, weight_obstacle,
                        obstacle_order):
    """
    Analyzes a prediction file against the ground truth.

    Args:
        result_filepath (str): Path to the model's prediction CSV file.
        ground_truth_filepath (str): Path to the ground truth CSV file.
        output_filepath (str): Path to save the output report file.
    """
    print(f"Analyzing {result_filepath}...")
    try:
        # Create output directory based on the result file name
        output_dir = os.path.dirname(output_filepath)
        os.makedirs(output_dir, exist_ok=True)

        # Load data
        gt_df = pd.read_csv(ground_truth_filepath)
        pred_df = pd.read_csv(result_filepath)

        # Merge ground truth and predictions
        merged_df = pd.merge(gt_df, pred_df, on='file_name', suffixes=('_gt', '_pred'))

        # Dynamically determine the categories to analyze
        categories = [col for col in gt_df.columns if col != 'file_name']
        
        # Use a string buffer to build the report
        report_buffer = io.StringIO()
        category_scores = {}

        for category in categories:
            gt_col = f'{category}_gt'
            pred_col = f'{category}_pred'

            if gt_col not in merged_df.columns or pred_col not in merged_df.columns:
                report_buffer.write(f"Category '{category}' not found. Skipping.\n\n")
                continue

            y_true = merged_df[gt_col]
            y_pred = merged_df[pred_col]
            
            if category == 'Obstacle':
                labels = obstacle_order
            else:
                labels = sorted(list(set(y_true.unique()) | set(y_pred.unique())))

            cm = confusion_matrix(y_true, y_pred, labels=labels)
            cm_df = pd.DataFrame(cm, index=labels, columns=labels)

            report_buffer.write(f"--- Analysis for Category: {category} ---\n\n")
            report_buffer.write("Confusion Matrix:\n")
            report_buffer.write(cm_df.to_string())
            report_buffer.write("\n\n")

            metrics_df = calculate_precision_recall(cm, labels)
            report_buffer.write("Metrics per Sub-category:\n")
            report_buffer.write(metrics_df.to_string(index=False))
            report_buffer.write("\n\n")

            if category == 'decision':
                f_beta = fbeta_score(y_true, y_pred, beta=beta_decision, average='macro', labels=labels, zero_division=0)
                category_scores['decision_fbeta'] = f_beta
                report_buffer.write(f"Macro F-{beta_decision} Score: {f_beta:.4f}\n")
            elif category == 'lane':
                f_beta = fbeta_score(y_true, y_pred, beta=beta_lane, average='macro', labels=labels, zero_division=0)
                category_scores['lane_fbeta'] = f_beta
                report_buffer.write(f"Macro F-{beta_lane} Score: {f_beta:.4f}\n")
            elif category == 'obstacle':
                f_beta = fbeta_score(y_true, y_pred, beta=beta_obstacle, average='macro', labels=labels, zero_division=0)
                kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic', labels=labels)
                category_scores['obstacle_fbeta'] = f_beta
                category_scores['obstacle_kappa'] = kappa
                report_buffer.write(f"Macro F-{beta_obstacle} Score: {f_beta:.4f}\n")
                report_buffer.write(f"Quadratic Weighted Cohen's Kappa: {kappa:.4f}\n")

            report_buffer.write("\n" + "="*50 + "\n\n")

        # --- Overall Metrics ---
        report_buffer.write("--- Overall Performance Metrics ---\n\n")

        # Complete Exact Accuracy
        correct_predictions = (merged_df[[f'{cat}_gt' for cat in categories]].values == merged_df[[f'{cat}_pred' for cat in categories]].values).all(axis=1).sum()
        complete_exact_accuracy = correct_predictions / len(merged_df)
        report_buffer.write(f"Complete Exact Accuracy: {complete_exact_accuracy:.4f} ({correct_predictions}/{len(merged_df)})\n\n")

        # Composite Score
        decision_fbeta = category_scores.get('decision_fbeta', 0)
        lane_fbeta = category_scores.get('lane_fbeta', 0)
        obstacle_kappa = category_scores.get('obstacle_kappa', 0)
        obstacle_fbeta = category_scores.get('obstacle_fbeta', 0)

        composite_score = (
            decision_fbeta * weight_decision +
            lane_fbeta * weight_lane +
            obstacle_kappa * weight_obstacle
        )
        
        report_buffer.write("--- Components of Composite Score ---\n")
        report_buffer.write(f"Decision Macro F-{beta_decision} Score: {decision_fbeta:.4f}\n")
        report_buffer.write(f"Lane Macro F-{beta_lane} Score: {lane_fbeta:.4f}\n")
        report_buffer.write(f"Obstacle Macro F-{beta_obstacle} Score: {obstacle_fbeta:.4f}\n")
        report_buffer.write(f"Obstacle Quadratic Weighted Cohen's Kappa: {obstacle_kappa:.4f}\n\n")

        report_buffer.write(f"Composite Score: {composite_score:.4f}\n")
        report_buffer.write(f"(Weights: Decision F-beta={weight_decision}, Lane F-beta={weight_lane}, Obstacle Kappa={weight_obstacle})\n")
        
        # Save the report
        with open(output_filepath, 'w') as f:
            f.write(report_buffer.getvalue())

        print(f"Analysis complete. Report saved to: {output_filepath}")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please check your file paths.")
    except Exception as e:
        print(f"An unexpected error occurred while analyzing {result_filepath}: {e}")


def main():
    """
    Main function to define file paths and run the analysis.
    """
    parser = argparse.ArgumentParser(description="Analyze model prediction results against ground truth.")
    parser.add_argument("input_dir", help="The root directory containing model run subdirectories.")
    args = parser.parse_args()
    
    # --- User Configuration ---
    # F-beta score parameters (beta > 1 gives more weight to recall, beta < 1 to precision)
    BETA_DECISION = 2.0  # Emphasize recall for Decision
    BETA_LANE = 1.0      # Balanced F1-score for Lane
    BETA_OBSTACLE = 1.0  # Balanced F1-score for Obstacle

    # Composite score weights
    WEIGHT_DECISION = 0.5
    WEIGHT_LANE = 0.3
    WEIGHT_OBSTACLE = 0.2

    # Ordinal labels for Obstacle category for Cohen's Kappa.
    # The order is important: from most to least critical.
    OBSTACLE_ORDER = ["very close", "near", "far away", "not on the same lane"]
    # --------------------------

    try:
        subdirs = [f.path for f in os.scandir(args.input_dir) if f.is_dir()]
    except FileNotFoundError:
        print(f"Error: Input directory not found at {args.input_dir}")
        return

    for run_dir in subdirs:
        run_name = os.path.basename(run_dir)
        print(f"--- Processing Run: {run_name} ---")
        
        for type_ in ['real', 'simulation']:
            suffix = 'real' if type_ == 'real' else 'sim'
            pred_file = os.path.join(run_dir, 'extraction_results', type_, f'{run_name}_{suffix}.csv')
            gt_file = os.path.join(run_dir, 'extraction_results', type_, f'ground_truth_{type_}.csv')
            
            output_dir = os.path.join(run_dir, 'categorical_comparison')
            output_file = os.path.join(output_dir, f'{run_name}_{type_}_report.txt')

            if os.path.exists(pred_file) and os.path.exists(gt_file):
                analyze_predictions(
                    pred_file, gt_file, output_file,
                    BETA_DECISION, BETA_LANE, BETA_OBSTACLE,
                    WEIGHT_DECISION, WEIGHT_LANE, WEIGHT_OBSTACLE,
                    OBSTACLE_ORDER
                )
            else:
                print(f"Skipping {type_} for {run_name} - files not found.")
                if not os.path.exists(pred_file):
                    print(f"  Missing: {pred_file}")
                if not os.path.exists(gt_file):
                    print(f"  Missing: {gt_file}")
        print("-" * (24 + len(run_name)) + "\n")


if __name__ == "__main__":
    main() 