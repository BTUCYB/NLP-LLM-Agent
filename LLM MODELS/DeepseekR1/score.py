import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    confusion_matrix,
    matthews_corrcoef,
    roc_curve,
    precision_recall_curve
)

def detection_rate(y_true, y_pred):
    return recall_score(y_true, y_pred)

def false_positive_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0

def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def balanced_accuracy(y_true, y_pred):
    return 0.5 * (recall_score(y_true, y_pred) + specificity(y_true, y_pred))

def generate_evaluation_plots(y_true, y_probs, model_name):
    """One figure at a time: ROC, PR, CM, F1 vs Threshold, MCC vs Threshold, Log-Loss vs Threshold."""
    
    # 1) ROC Curve
    plt.figure(figsize=(6,5))
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.plot(fpr, tpr, lw=2, label=f'AUC={roc_auc_score(y_true, y_probs):.2f}')
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.title(f'{model_name} ROC Curve')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right'); plt.tight_layout(); plt.show()

    # 2) Precision–Recall Curve
    plt.figure(figsize=(6,5))
    prec, rec, _ = precision_recall_curve(y_true, y_probs)
    plt.plot(rec, prec, lw=2, label=f'PR AUC={average_precision_score(y_true, y_probs):.2f}')
    plt.title(f'{model_name} Precision–Recall Curve')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.legend(loc='upper right'); plt.tight_layout(); plt.show()

    # 3) Confusion Matrix @ 0.5
    plt.figure(figsize=(6,5))
    y_pred_class = (y_probs >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred_class)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal','Threat'],
                yticklabels=['Normal','Threat'])
    plt.title(f'{model_name} Confusion Matrix (0.5)')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.tight_layout(); plt.show()

    # Prepare thresholds array just once
    thresholds = np.linspace(0, 1, 100)

    # 4) Threshold → F1 Score
    plt.figure(figsize=(6,4))
    f1_scores = [f1_score(y_true, (y_probs >= t).astype(int)) for t in thresholds]
    plt.plot(thresholds, f1_scores, lw=2)
    plt.title(f'{model_name} Threshold → F1 Score')
    plt.xlabel('Threshold'); plt.ylabel('F1 Score')
    plt.grid(True, linestyle='--', alpha=0.5); plt.tight_layout(); plt.show()

    # 5) Threshold → MCC Score
    plt.figure(figsize=(6,4))
    mcc_scores = [matthews_corrcoef(y_true, (y_probs >= t).astype(int)) for t in thresholds]
    plt.plot(thresholds, mcc_scores, lw=2)
    plt.title(f'{model_name} Threshold → MCC')
    plt.xlabel('Threshold'); plt.ylabel('Matthews Corrcoef')
    plt.grid(True, linestyle='--', alpha=0.5); plt.tight_layout(); plt.show()

    # 6) Threshold → Log Loss
    plt.figure(figsize=(6,4))
    # avoid 0 or 1 exactly to prevent infinite log loss
    thresh_for_loss = np.linspace(0.01, 0.99, 99)
    log_losses = []
    for t in thresh_for_loss:
        # simulate hardening probabilities to exactly t or 1−t
        p_est = np.where(y_probs >= t, t, 1 - t)
        log_losses.append(log_loss(y_true, p_est))
    plt.plot(thresh_for_loss, log_losses, lw=2)
    plt.title(f'{model_name} Threshold → Log Loss')
    plt.xlabel('Threshold'); plt.ylabel('Log Loss')
    plt.grid(True, linestyle='--', alpha=0.5); plt.tight_layout(); plt.show()

def generate_evaluation_dataframes(y_true, y_probs):
    """Generate DataFrames containing evaluation curve data"""
    # ROC Curve
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_probs)
    df_roc = pd.DataFrame({
        'Threshold': thresholds_roc,
        'FPR (False Positive Rate)': fpr,
        'TPR (True Positive Rate)': tpr
    }).sort_values('Threshold', ascending=False)
    
    # Precision-Recall Curve
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_probs)
    df_pr = pd.DataFrame({
        'Threshold': np.append(thresholds_pr, 0),  # Add final threshold=0
        'Precision': precision,
        'Recall': recall
    })
    
    # Threshold Analysis
    thresholds = np.linspace(0, 1, 100)
    df_thresholds = pd.DataFrame({
        'Threshold': thresholds,
        'F1': [f1_score(y_true, (y_probs >= t).astype(int)) for t in thresholds],
        'MCC': [matthews_corrcoef(y_true, (y_probs >= t).astype(int)) for t in thresholds]
    })
    
    # Log Loss Analysis
    thresh_logloss = np.linspace(0.01, 0.99, 99)
    log_losses = []
    for t in thresh_logloss:
        p_est = np.where(y_probs >= t, t, 1 - t)
        log_losses.append(log_loss(y_true, p_est))
    df_logloss = pd.DataFrame({
        'Threshold': thresh_logloss,
        'Log Loss': log_losses
    })
    
    return {
        'ROC_Curve': df_roc,
        'PR_Curve': df_pr,
        'Threshold_Analysis': df_thresholds,
        'LogLoss_Analysis': df_logloss
    }

def evaluate_threat_models(y_true, y_pred, y_probs, model_name="Model"):
    """Full evaluation with metrics, plots, and DataFrames"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # Calculate metrics
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall (Detection Rate)': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'ROC AUC': roc_auc_score(y_true, y_probs),
        'Log Loss': log_loss(y_true, y_probs),
       
    }

    # Generate plots
    generate_evaluation_plots(y_true, y_probs, model_name)
    
    # Generate DataFrames
    dfs = generate_evaluation_dataframes(y_true, y_probs)
    
    return metrics, dfs

def generate_score_explanation(metric_name, score):
    """Generate context-specific explanations based on score value"""
    explanations = {
        'Accuracy': (
            f"{(score*100):.1f}% correct overall predictions. For insider threats, values <70% indicate "
            "significant detection gaps, as even random guessing might perform better in imbalanced scenarios."
        ),
        'Precision': (
            f"{(score*100):.1f}% of flagged users are actual threats. "
            "Security teams can tolerate lower values (40-60%) if recall is high, "
            "but <30% indicates excessive false alarms."
        ),
        'Recall (Detection Rate)': (
            f"Detects {(score*100):.1f}% of real threats. Critical security metric - "
            "values <80% leave dangerous gaps, <60% indicates fundamental detection flaws."
        ),
        'F1 Score': (
            f"Balanced performance score: {score:.2f}. "
            "Aim for >0.7 in production systems. "
            f"{'Needs immediate improvement' if score < 0.5 else 'Marginally acceptable' if score < 0.7 else 'Good performance'} "
            "for threat detection."
        ),
        'ROC AUC': (
            f"Overall detection capability: {score:.2f}. "
            "0.9+ = Excellent, 0.8-0.9 = Good, <0.7 = Needs model retuning. "
            "Indicates how well threats are prioritized over normal users."
        ),
        
        'Log Loss': (
            f"Prediction confidence quality: {score:.2f}. "
            "Lower is better. <0.5 indicates good certainty in predictions, "
            ">1.0 suggests the model struggles to distinguish threats."
        ),
        
    }
    return explanations.get(metric_name, "No interpretation available")

def print_metric_table(metrics):
    """Print metrics in a formatted table with score-specific explanations"""
    # Create DataFrame with dynamic explanations
    df = pd.DataFrame([
        {
            'Attribute': name,
            'Score': f"{val:.2%}" if any(x in name for x in ['Rate', 'Specificity', 'Accuracy']) else f"{val:.4f}",
            'Interpretation': generate_score_explanation(name, val)
        }
        for name, val in metrics.items()
    ])
    
    # Print table with styling
    print("\n=== Insider Threat Detection Evaluation Metrics ===")
    print(df.to_string(index=False, justify='left', max_colwidth=60))

if __name__ == "__main__":
    # Example data (replace with your model outputs)
    y_true = [1,0,1,0,1,0,1,1,0,1,0,1,0,1,0]
    y_pred = [1,1,1,0,0,0,1,1,0,1,1,1,0,1,0]
    y_prob = [0.7,0.6,0.65,0.4,0.45,0.3,0.8,0.75,0.35,0.72,0.55,0.68,0.38,0.81,0.28]

    # Run evaluation
    metrics, dataframes = evaluate_threat_models(
        y_true, y_pred, y_prob, model_name="DeepseekV1 Threat Model"
    )

    # Print formatted table
    print_metric_table(metrics)

    # Show DataFrame examples
    print("\n=== Example DataFrame Outputs ===")
    print("ROC Curve Data (First 5 rows):")
    print(dataframes['ROC_Curve'].head())
    
    print("\nThreshold Analysis Data (First 5 rows):")
    print(dataframes['Threshold_Analysis'].head())

    # Example: Find optimal threshold
    df = dataframes['Threshold_Analysis']
    optimal_row = df.loc[df['MCC'].idxmax()]
    print(f"\nOptimal Threshold: {optimal_row['Threshold']:.2f}")
    print(f"At this threshold:")
    print(f"- MCC: {optimal_row['MCC']:.2f}")
    print(f"- F1: {optimal_row['F1']:.2f}")