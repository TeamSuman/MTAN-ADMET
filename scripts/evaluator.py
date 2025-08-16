import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, roc_auc_score, roc_curve
import matplotlib.patheffects as pe
import pandas as pd

class MultiTaskEvaluator:
    def __init__(self, label, binary, benchmark):
        self.label = label
        self.binary = binary
        self.benchmark = benchmark
        self.results = []

    def evaluate(self, train_pred, train_label, val_pred, val_label, test_pred, test_label):
        for i, task in enumerate(self.label):
            if self.binary[i]:
                # Binary classification: AUC + ROC curves
                scores = {}
                fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

                for split_name, preds, labels, color in [
                    ("Train", train_pred, train_label, "blue"),
                    ("Val",   val_pred,   val_label,   "green"),
                    ("Test",  test_pred,  test_label,  "red")
                ]:
                    y_true = labels[:, i]
                    y_pred = torch.sigmoid(torch.tensor(preds[:, i])).numpy()
                    mask = ~np.isnan(y_true)

                    if mask.sum() > 0 and len(np.unique(y_true[mask])) > 1:
                        auc = roc_auc_score(y_true[mask], y_pred[mask])
                        fpr, tpr, _ = roc_curve(y_true[mask], y_pred[mask])
                        ax.plot(fpr, tpr, label=f"{split_name} AUC={auc:.3f}", color=color)
                        scores[split_name] = auc
                    else:
                        scores[split_name] = np.nan

                ax.plot([0, 1], [0, 1], 'k--')
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title(f"ROC Curve - {task}")
                ax.legend()
                plt.show()

                self.results.append({
                    "Task": task,
                    "Type": "Binary",
                    "Train Score": scores["Train"],
                    "Val Score": scores["Val"],
                    "Test Score": scores["Test"],
                    "Benchmark": self.benchmark[i]
                })

            else:
                # Regression: R², MAE, RMSE + Predicted vs Actual plot
                scores = {}
                fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

                for split_name, preds, labels, color in [
                    ("Train", train_pred, train_label, "blue"),
                    ("Val",   val_pred,   val_label,   "green"),
                    ("Test",  test_pred,  test_label,  "red")
                ]:
                    y_true = labels[:, i]
                    y_pred = preds[:, i]
                    mask = ~np.isnan(y_true)

                    if mask.sum() > 0:
                        r2 = r2_score(y_true[mask], y_pred[mask])
                        mae = mean_absolute_error(y_true[mask], y_pred[mask])
                        rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
                        ax.scatter(y_true[mask], y_pred[mask], label=f"{split_name} R²={r2:.3f}", color=color, alpha=0.6)
                        scores[f"{split_name} R²"] = r2
                        scores[f"{split_name} MAE"] = mae
                        scores[f"{split_name} RMSE"] = rmse
                    else:
                        scores[f"{split_name} R²"] = np.nan
                        scores[f"{split_name} MAE"] = np.nan
                        scores[f"{split_name} RMSE"] = np.nan

                ax.plot(ax.get_xlim(), ax.get_xlim(), 'k--')
                ax.set_xlabel(f"Actual {task}")
                ax.set_ylabel(f"Predicted {task}")
                ax.set_title(f"Regression Plot - {task}")
                ax.legend()
                plt.show()

                self.results.append({
                    "Task": task,
                    "Type": "Regression",
                    **scores,
                    "Benchmark": self.benchmark[i]
                })

    def get_results_table(self):
        return pd.DataFrame(self.results)
