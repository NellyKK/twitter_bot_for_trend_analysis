import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load saved results
results = pd.read_csv(r"D:\Advanced_Python\project\fold_results.csv")

fold_accuracies = results['Accuracy'].values
fold_losses = results['Loss'].values
NUM_FOLDS = len(results)

# Print summary
print("=" * 50)
print("10-FOLD CROSS VALIDATION RESULTS")
print("=" * 50)
for i in range(NUM_FOLDS):
    print(f"  Fold {i+1:2d}: Accuracy = {fold_accuracies[i]:.4f}, Loss = {fold_losses[i]:.4f}")
print("-" * 50)
print(f"  Mean Accuracy:  {np.mean(fold_accuracies):.4f}")
print(f"  Std Accuracy:   {np.std(fold_accuracies):.4f}")
print(f"  Mean Loss:      {np.mean(fold_losses):.4f}")
print("=" * 50)

# Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(range(1, NUM_FOLDS + 1), fold_accuracies, color='steelblue')
plt.axhline(y=np.mean(fold_accuracies), color='red', linestyle='--', label=f'Mean: {np.mean(fold_accuracies):.4f}')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Accuracy per Fold')
plt.xticks(range(1, NUM_FOLDS + 1))
plt.legend()

plt.subplot(1, 2, 2)
plt.bar(range(1, NUM_FOLDS + 1), fold_losses, color='coral')
plt.axhline(y=np.mean(fold_losses), color='red', linestyle='--', label=f'Mean: {np.mean(fold_losses):.4f}')
plt.xlabel('Fold')
plt.ylabel('Loss')
plt.title('Loss per Fold')
plt.xticks(range(1, NUM_FOLDS + 1))
plt.legend()

plt.tight_layout()
plt.show()