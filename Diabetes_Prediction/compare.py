import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Full path to project root
base_path = "/Users/shadmansakib/Documents/Diabetes_Prediction/Diabetes_Prediction"

# Load ground truth
ground_truth_df = pd.read_csv(os.path.join(base_path, "diabetes.csv"))
true_labels = ground_truth_df["Outcome"].values

# Define absolute prediction file paths
prediction_paths = [
    os.path.join(base_path, "models/mistral/diabetes_predictions_mistral_one_shot.csv"),
    os.path.join(base_path, "models/mistral/diabetes_predictions_mistral_three_shot.csv"),
    os.path.join(base_path, "models/mistral/diabetes_predictions_mistral_zero_shot.csv"),
    os.path.join(base_path, "models/llama3.2/diabetes_predictions_llama3.2_zero_shot.csv"),
    os.path.join(base_path, "models/llama3.2/diabetes_predictions_llama3.2_three_shot.csv"),
    os.path.join(base_path, "models/llama3.2/diabetes_predictions_llama3.2_one_shot.csv"),
    os.path.join(base_path, "models/llama3.1/diabetes_predictions_llama3.1_zero_shot.csv"),
    os.path.join(base_path, "models/llama3.1/diabetes_predictions_llama3.1_three_shot.csv"),
    os.path.join(base_path, "models/llama3.1/diabetes_predictions_llama3.1_one_shot.csv"),
    os.path.join(base_path, "models/gemma/diabetes_predictions_gemma2_zero_shot.csv"),
    os.path.join(base_path, "models/gemma/diabetes_predictions_gemma2_three_shot.csv"),
    os.path.join(base_path, "models/gemma/diabetes_predictions_gemma2_one_shot.csv"),
    os.path.join(base_path, "models/gemini/diabetes_predictions_gemini_zero_shot.csv"),
    os.path.join(base_path, "models/gemini/diabetes_predictions_gemini_three_shot.csv"),
    os.path.join(base_path, "models/gemini/diabetes_predictions_gemini_one_shot.csv"),
    os.path.join(base_path, "models/chatgpt/diabetes_predictions_chatgpt_three_shot.csv"),
    os.path.join(base_path, "models/chatgpt/diabetes_predictions_chatgpt_zero_shot.csv"),
    os.path.join(base_path, "models/chatgpt/diabetes_predictions_chatgpt_one_shot.csv"),
]

# Output collection
results = []
misclassified_all = []

for path in prediction_paths:
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        continue

    model_name = os.path.basename(path).replace("diabetes_predictions_", "").replace(".csv", "")
    preds = pd.read_csv(path).iloc[:, 0].values

    report = classification_report(true_labels, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(true_labels, preds)

    misclassified_mask = (true_labels != preds)
    misclassified = ground_truth_df[misclassified_mask].copy()
    misclassified["True_Label"] = true_labels[misclassified_mask]
    misclassified["Predicted_Label"] = preds[misclassified_mask]
    misclassified["Model"] = model_name
    misclassified_all.append(misclassified)

    results.append({
        "Model": model_name,
        "Accuracy": report["accuracy"],
        "Precision (1)": report["1"]["precision"],
        "Recall (1)": report["1"]["recall"],
        "F1-Score (1)": report["1"]["f1-score"],
        "False Positives": cm[0][1],
        "False Negatives": cm[1][0],
        "Total Errors": misclassified.shape[0]
    })

# Save outputs if any
if not results:
    print("❌ No valid prediction files found. Please check paths.")
else:
    summary_df = pd.DataFrame(results)
    summary_df.sort_values(by="F1-Score (1)", ascending=False, inplace=True)
    summary_path = os.path.join(base_path, "llm_misclassification_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ Saved summary: {summary_path}")

    if misclassified_all:
        misclassified_combined = pd.concat(misclassified_all, ignore_index=True)
        error_path = os.path.join(base_path, "llm_misclassified_samples.csv")
        misclassified_combined.to_csv(error_path, index=False)
        print(f"✅ Saved misclassified samples: {error_path}")
