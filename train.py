import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import SwitchTransformer  # Update this import
from utils import load_data, save_model, evaluate_model

def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load datasets
    train_data = load_data(os.path.join('DatasetsForCyberThreat', 'train_data.csv'))
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    # Initialize model, criterion, and optimizer
    model = SwitchTransformer(num_experts=4, input_dim=100, output_dim=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs=10, device=device)

    # Save the trained model
    save_model(model, 'switch_transformer_model.pth')

    # Optionally evaluate the model
    evaluate_model(model, train_loader, device)

if __name__ == "__main__":
    main()