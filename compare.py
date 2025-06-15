import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# --- PATH SETUP ---

# Full path to your ground truth CSV file
ground_truth_path = "/Users/shadmansakib/Documents/Diabetes_Prediction/data/raw/diabetes.csv"

# Base path where your model folders (mistral/, llama3.1/, etc.) are located
predictions_base_path = "/Users/shadmansakib/Documents/Diabetes_Prediction/models"

# --- LOAD GROUND TRUTH ---

ground_truth_df = pd.read_csv(ground_truth_path)
true_labels = ground_truth_df["Outcome"].astype(int).values
num_samples = len(true_labels)

# --- PREDICTION FILE PATHS ---

models = ["mistral", "llama3.2", "llama3.1", "gemma2", "gemini", "chatgpt"]
prompt_types = ["zero_shot", "one_shot", "three_shot"]

prediction_paths = [
    os.path.join(predictions_base_path, model, f"diabetes_predictions_{model}_{prompt}.csv")
    for model in models
    for prompt in prompt_types
]

# --- FUNCTION TO CLEAN PREDICTIONS ---

def load_and_clean_predictions(path):
    df = pd.read_csv(path)
    # Extract only first digit (0 or 1) from each prediction
    preds_raw = df.iloc[:, 0].astype(str)
    cleaned = preds_raw.str.extract(r"(\d)").fillna("0").astype(int)[0]
    return cleaned[:num_samples].values

# --- ANALYSIS LOOP ---

results = []
misclassified_all = []

for path in prediction_paths:
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        continue

    model_name = os.path.basename(path).replace("diabetes_predictions_", "").replace(".csv", "")
    preds = load_and_clean_predictions(path)

    if len(preds) != num_samples:
        print(f"[ERROR] Prediction count mismatch for {model_name} (got {len(preds)}, expected {num_samples})")
        continue

    report = classification_report(true_labels, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(true_labels, preds)

    misclassified_mask = true_labels != preds
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

# --- OUTPUT RESULTS ---

if not results:
    print("❌ No valid prediction files found or processed.")
else:
    summary_df = pd.DataFrame(results)
    summary_df.sort_values(by="F1-Score (1)", ascending=False, inplace=True)

    summary_path = os.path.join(predictions_base_path, "llm_misclassification_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ Saved summary: {summary_path}")

    if misclassified_all:
        misclassified_combined = pd.concat(misclassified_all, ignore_index=True)
        error_path = os.path.join(predictions_base_path, "llm_misclassified_samples.csv")
        misclassified_combined.to_csv(error_path, index=False)
        print(f"✅ Saved misclassified samples: {error_path}")
