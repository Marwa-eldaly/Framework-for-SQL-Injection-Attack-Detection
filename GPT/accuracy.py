import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import argparse
import pip
pip.main(['install','seaborn'])
import seaborn as sns
import matplotlib.pyplot as plt



def calculate_metrics(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    # print(df.head())
    # Clean classifier column and map to 1/0
    df['classification'] = df['classification'].str.lower().str.replace('.', '').map({'yes': 1, 'no': 0})
    
    # Get true labels and predictions
    y_true = df['Label']
    y_pred = df['classification']
    
    # Check for nulls in y_pred
    null_mask = y_pred.isnull()
    if null_mask.any():
        print(f"Found {null_mask.sum()} null values in predictions")
        print("Rows with null predictions:")
        print(df[null_mask])
    
    # print(y_true.value_counts())
    # Calculate basic counts    
    # return

    total_instances = len(df)
    total_positives = sum(y_true == 1)
    total_negatives = sum(y_true == 0)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Create confusion matrix visualization
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Vulnerable', 'Vulnerable'],
                yticklabels=['Not Vulnerable', 'Vulnerable'])
    plt.title('Confusion Matrix for gpt-4 Model ')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Print results
    print(f"Total Instances: {total_instances}")
    print(f"Total Positives: {total_positives}")
    print(f"Total Negatives: {total_negatives}")
    print(f"True Positives (TP): {tp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-score: {f1 * 100:.2f}%")
    
    # Save metrics to a new CSV file
    metrics_df = pd.DataFrame({
        'Metric': ['Total Instances', 'Total Positives', 'Total Negatives', 
                  'True Positives', 'True Negatives', 'False Positives', 
                  'False Negatives', 'Accuracy', 'Precision', 'Recall', 'F1-score'],
        'Value': [total_instances, total_positives, total_negatives, 
                 tp, tn, fp, fn, accuracy * 100, precision * 100, recall * 100, f1 * 100]
    })
    
    metrics_df.to_csv('metrics_results.csv', index=False)

    # Create bar chart for metrics
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    metrics_values = [accuracy * 100, precision * 100, recall * 100, f1 * 100]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics_names, metrics_values, color=['#2ecc71', '#3498db', '#9b59b6', '#f1c40f'])
    plt.title('Model Performance Metrics', pad=20)
    plt.ylabel('Percentage (%)')
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom')
    
    # Set y-axis to go from 0 to 100
    plt.ylim(0, 100)
    
    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('performance_metrics.png')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate classification metrics from CSV file')
    parser.add_argument('csv_file', help='Path to the CSV file')
    args = parser.parse_args()
    
    calculate_metrics(args.csv_file)
