"""
This file contains basics helper function for pyTorch models used by colab NB (e.g. conversion of numpy arrays to tensors, building out training and testing loops, etc.)
"""
import torch
from sklearn.metrics import accuracy_score


def to_tensor(arr):
    """
    Convert a numpy array to a pyTorch tensor
    """
    return torch.tensor(arr, dtype=torch.float32)


def to_cpu(tensor):
    """
    Move a tensor to the CPU
    """
    return tensor.cpu()


def training_testing_loop_classification_model(model, loss_fn, optimizer, epochs, X_train, y_train, X_test, y_test, printStep=100):

    for epoch in range(epochs):
        # training
        model.train()

        # 1. Forward pass
        y_logits = model(X_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))

        # 2. Compute loss
        loss = loss_fn(y_logits, y_train)
        acc = accuracy_score(y_train, y_pred.detach().numpy())

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Backward pass
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Testing
        model.eval()
        with torch.inference_mode():
            # 1. forward pass
            test_logits = model(X_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))

            # 2. Compute loss
            test_loss = loss_fn(test_logits, y_test)
            test_acc = accuracy_score(y_test, test_pred.detach().numpy())

        # printing the results
        if printStep % 100 == 0:
            print(
                f"Epoch {epochs} | Training Loss: {loss:.4f}  Training Acc: {acc:.4f}% | Testing Loss: {test_loss:.4f} Testing Acc: {test_acc:.4f}%")
