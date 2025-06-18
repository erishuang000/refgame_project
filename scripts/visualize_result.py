import json
import matplotlib.pyplot as plt
import os
import pandas as pd

# --- Configuration ---
RESULTS_FILE = "/puhome/23063003r/refgame_project/output/training_results.json"  # Path to the JSON results file
OUTPUT_VISUALIZE_DIR = "/puhome/23063003r/refgame_project/output_visualize/" # Directory to save plots
ROLLING_WINDOW = 100 # Window size for rolling average/sum for smoothing

# --- Create output directory if it doesn't exist ---
if not os.path.exists(OUTPUT_VISUALIZE_DIR):
    os.makedirs(OUTPUT_VISUALIZE_DIR)
    print(f"Created output directory: {OUTPUT_VISUALIZE_DIR}")

# --- Load data from JSON file ---
try:
    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    per_round_metrics = data.get('per_round_metrics', [])
    if not per_round_metrics:
        print(f"No 'per_round_metrics' found in {RESULTS_FILE}. Cannot generate plots.")
        exit()
except FileNotFoundError:
    print(f"Error: Results file '{RESULTS_FILE}' not found. Please ensure it exists.")
    exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from '{RESULTS_FILE}'. Check file integrity.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading data: {e}")
    exit()

# Convert to pandas DataFrame for easier processing
df = pd.DataFrame(per_round_metrics)

# --- Calculate rolling metrics ---
# Rolling average loss
df['rolling_avg_loss'] = df['final_loss'].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
# Rolling accuracy (convert boolean to int for sum, then to mean for accuracy)
df['rolling_accuracy'] = df['is_correct_prediction'].astype(int).rolling(window=ROLLING_WINDOW, min_periods=1).mean() * 100
# Rolling embedding diff norm (keep as is, or maybe rolling mean for smoother trend)
df['rolling_avg_embedding_diff'] = df['embedding_diff_norm'].rolling(window=ROLLING_WINDOW, min_periods=1).mean()


# --- Plotting ---

# Plot 1: Loss over rounds
plt.figure(figsize=(12, 6))
plt.plot(df['round_idx'], df['rolling_avg_loss'], label=f'Rolling Average Loss ({ROLLING_WINDOW} rounds)', color='red')
plt.xlabel('Training Rounds')
plt.ylabel('Loss')
plt.title('Average Loss Over Training Rounds')
plt.grid(True)
plt.legend()
plt.tight_layout()
loss_plot_path = os.path.join(OUTPUT_VISUALIZE_DIR, 'average_loss_over_rounds.png')
plt.savefig(loss_plot_path)
plt.close()
print(f"Saved plot: {loss_plot_path}")

# Plot 2: Accuracy over rounds
plt.figure(figsize=(12, 6))
plt.plot(df['round_idx'], df['rolling_accuracy'], label=f'Rolling Accuracy ({ROLLING_WINDOW} rounds)', color='blue')
plt.xlabel('Training Rounds')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Over Training Rounds')
plt.grid(True)
plt.legend()
plt.tight_layout()
accuracy_plot_path = os.path.join(OUTPUT_VISUALIZE_DIR, 'accuracy_over_rounds.png')
plt.savefig(accuracy_plot_path)
plt.close()
print(f"Saved plot: {accuracy_plot_path}")

# Plot 3: Embedding change norm over rounds
plt.figure(figsize=(12, 6))
plt.plot(df['round_idx'], df['embedding_diff_norm'], label='Embedding Change Norm (Per Round)', color='green', alpha=0.6)
plt.plot(df['round_idx'], df['rolling_avg_embedding_diff'], label=f'Rolling Average Embedding Change ({ROLLING_WINDOW} rounds)', color='darkgreen', linestyle='--')
plt.xlabel('Training Rounds')
plt.ylabel('L2 Norm of Change')
plt.title('Embedding Change Norm Over Training Rounds')
plt.grid(True)
plt.legend()
plt.tight_layout()
embedding_diff_plot_path = os.path.join(OUTPUT_VISUALIZE_DIR, 'embedding_change_over_rounds.png')
plt.savefig(embedding_diff_plot_path)
plt.close()
print(f"Saved plot: {embedding_diff_plot_path}")
