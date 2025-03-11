import torch
import numpy as np
import matplotlib.pyplot as plt

def quantile_loss(y, y_hat, quantiles):
    """
    Compute the basic quantile loss.
    Args:
        y (Tensor): Targets of shape [batch_size, forecast_horizon, 1].
        y_hat (Tensor): Predictions of shape [batch_size, len(quantiles), forecast_horizon].
        quantiles (list): List of quantiles.
    Returns:
        total_loss (Tensor): Combined quantile loss (scalar).
    """
    losses = []

    # Reshape targets to [batch_size, 1, forecast_horizon] for broadcasting
    y = y.permute(0, 2, 1)  # [batch_size, 1, forecast_horizon]

    for i, q in enumerate(quantiles):
        # Compute quantile loss for each quantile
        errors = y - y_hat[:, i, :].unsqueeze(1)  # [batch_size, 1, forecast_horizon]
        quantile_loss = torch.max(q * errors, (q - 1) * errors).mean()
        losses.append(quantile_loss)

    # Combine all quantile losses
    total_loss = torch.sum(torch.stack(losses))

    return total_loss



def predict_model(model, test_loader, target_scaler, quantiles, forecast_horizon, device, spaced=False):
    """
    Predict using the model on the test dataset.
    
    Args:
        model: Trained model for prediction.
        test_loader: DataLoader containing the test dataset.
        target_scaler: Scaler used to scale the target values (output_data).
        quantiles: List of quantiles for the predictions.
        forecast_horizon: Number of timesteps in the forecast horizon.
        device: Torch device to use for computation (CPU/GPU).
        spaced (bool): Whether the model is a spaced model (True) or simple model (False).
    
    Returns:
        forecast_inv: Inverse scaled forecasts with shape [num_samples, len(quantiles), forecast_horizon].
        true_inv: Inverse scaled true values with shape [num_samples, forecast_horizon].
    """
    model.eval()
    forecasts, true_values = [], []

    with torch.no_grad():
        for batch in test_loader:
            # Separate inputs based on model type
            if spaced:
                dense_inputs, spaced_inputs, future_inputs, pca_inputs, targets = batch
                dense_inputs, spaced_inputs, future_inputs, pca_inputs = (
                    dense_inputs.to(device),
                    spaced_inputs.to(device),
                    future_inputs.to(device),
                    pca_inputs.to(device),
                )
                preds = model(dense_inputs, spaced_inputs, future_inputs, pca_inputs)
            else:
                dense_inputs, future_inputs, pca_inputs, targets = batch
                dense_inputs, future_inputs, pca_inputs = dense_inputs.to(device), future_inputs.to(device), pca_inputs.to(device)
                preds = model(dense_inputs, future_inputs, pca_inputs)

            forecasts.append(preds.cpu())
            true_values.append(targets)

    # Convert forecasts and true_values to NumPy arrays
    forecasts = torch.cat(forecasts, dim=0).numpy()  # Shape: [num_samples, len(quantiles), forecast_horizon]
    true_values = torch.cat(true_values, dim=0).numpy()  # Shape: [num_samples, forecast_horizon, 1]

    # Reshape true_values for inverse scaling
    true_values_reshaped = true_values.reshape(-1, 1)  # Flatten for inverse scaling
    true_inv = target_scaler.inverse_transform(true_values_reshaped).reshape(-1, forecast_horizon)

    # Reshape forecasts for inverse scaling
    forecast_inv = []
    for q in range(len(quantiles)):
        scaled_forecast = forecasts[:, q, :].reshape(-1, 1)  # Flatten for inverse scaling
        forecast_inv.append(target_scaler.inverse_transform(scaled_forecast).reshape(-1, forecast_horizon))

    forecast_inv = np.stack(forecast_inv, axis=1)  # Combine quantiles into a single array

    return forecast_inv, true_inv

def plot_forecasts(forecasts, true_values, quantiles, forecast_horizon):
    """
    Visualizes the best and worst forecasts vs. true values based on MAE, pinball loss, and coverage.
    Args:
        forecasts (ndarray): Predicted values of shape [num_samples, num_quantiles, forecast_horizon].
        true_values (ndarray): True values of shape [num_samples, forecast_horizon].
        quantiles (list): List of quantiles corresponding to forecasts.
        forecast_horizon (int): Number of timesteps in the forecast horizon.
    """
    # Compute MAE for the median quantile
    mae = np.mean(np.abs(forecasts[:, 1, :] - true_values), axis=1)  # Median forecast

    # Compute pinball loss for all samples
    pinball_losses = []
    for i in range(forecasts.shape[0]):  # Loop over all samples
        pinball_losses.append(
            quantile_loss(
                torch.tensor(true_values[i:i+1, :, None]),
                torch.tensor(forecasts[i:i+1, :, :]),
                quantiles
            ).item()
        )

    # Compute coverage for the quantile range
    lower = forecasts[:, 0, :]
    upper = forecasts[:, 2, :]
    coverage = np.mean((true_values >= lower) & (true_values <= upper), axis=1)

    # Identify best and worst samples based on MAE
    best_idx = np.argmin(mae)
    worst_idx = np.argmax(mae)

    def plot_single_forecast(sample_idx, title):
        q_10, q_50, q_90 = forecasts[sample_idx, 0, :], forecasts[sample_idx, 1, :], forecasts[sample_idx, 2, :]
        true_vals = true_values[sample_idx, :]
        time_steps = np.arange(forecast_horizon)

        plt.figure(figsize=(12, 6))
        plt.fill_between(
            time_steps,
            q_10,
            q_90,
            color="gray",
            alpha=0.3,
            label=f"{quantiles[0]} - {quantiles[2]} Quantile Range",
        )
        plt.plot(time_steps, q_50, label=f"{quantiles[1]} Quantile (Median)", color="blue", linewidth=2)
        plt.plot(time_steps, true_vals, label="True Values", color="black", linestyle="--", linewidth=2)
        plt.title(title)
        plt.xlabel("Time Step")
        plt.ylabel("DA Value")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Print MAE, Pinball Loss, and Coverage
        print(f"MAE: {mae[sample_idx]:.4f}")
        print(f"Pinball Loss: {pinball_losses[sample_idx]:.4f}")
        print(f"Coverage: {coverage[sample_idx]:.4f}")

    # Plot best prediction
    plot_single_forecast(best_idx, f"Best Prediction (MAE: {mae[best_idx]:.4f})")

    # Plot worst prediction
    plot_single_forecast(worst_idx, f"Worst Prediction (MAE: {mae[worst_idx]:.4f})")

    # Print average MAE
    print(f"Average MAE: {np.mean(mae):.4f}")


def plot_training_validation_loss(train_losses, val_losses):
    """
    Plots training and validation losses.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
