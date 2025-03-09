import pandas as pd
import torch

def load_data(file_path):
    # Load data from a CSV file
    data = pd.read_csv(file_path)
    # Assuming the last column is the label
    inputs = data.iloc[:, :-1].values
    labels = data.iloc[:, -1].values
    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

def save_model(model, file_path):
    # Save the model's state dictionary
    torch.save(model.state_dict(), file_path)

def evaluate_model(model, test_loader, device):
    # Evaluate the model and calculate accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    return accuracy

def preprocess_data(raw_data):
    # Example: Normalize the data
    return (raw_data - raw_data.mean()) / raw_data.std()

def log_metrics(metrics):
    # Log the provided metrics
    for key, value in metrics.items():
        print(f'{key}: {value}')
        