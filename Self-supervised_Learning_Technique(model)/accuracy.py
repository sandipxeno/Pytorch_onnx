import torch
from self_supervised import device, dataloader
from supervised_learning_model_and_loss_function import model

# Function to compute accuracy
def calculate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == y).sum().item()
            total += y.size(0)

    accuracy = (correct / total) * 100
    return accuracy

# Check model accuracy
accuracy = calculate_accuracy(model, dataloader, device)
print(f"Model Accuracy: {accuracy:.2f}%")