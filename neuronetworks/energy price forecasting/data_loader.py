import pandas as pd 
import numpy as np
import glob
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class DataLoading:
    def __init__(self, file_path):
        self.file_path = file_path
    def load_data(self, file_path):
        """
        Load data from a CSV file.
        Args:
            file_path (str): Path to the CSV file containing the dataset.
        Returns:
            pd.DataFrame: Loaded dataset.
        """
        data = pd.read_csv(file_path)
        self.date_cols = ["Year", "Month", "Day", "Hour"]
        # One-hot encode the year, hour, day, month
        data["Weekday"] = pd.to_datetime(data["Year"].astype(str) + "-" + data["Month"].astype(str) + "-" + data["Day"].astype(str)).dt.weekday()
        data["Weekday"] = data["Weekday"].astype("category")
        self.temporal_cols = ["Year_scaled", "Hour_Sin", "Hour_Cos", "Day_Sin", "Day_Cos", "Month_Sin", "Month_Cos"]
        return data

    def create_features(input_features, target, dense_lookback, spaced_lookback, forecast_horizon, future_indices=None, step_growth_factor=None):
        """
        Create features with both dense and optionally spaced lookback for short- and long-term patterns.
        
        Args:
        - input_features (np.ndarray): Feature data.
        - target (np.ndarray): Target variable.
        - dense_lookback (int): Number of consecutive recent timesteps for dense lookback.
        - spaced_lookback (int): Maximum lookback period (e.g., 2 years). Set to 0 to skip spaced lookback.
        - forecast_horizon (int): Number of timesteps to forecast.
        - future_indices (list[int], optional): Indices of future features.
        - step_growth_factor (float, optional): If provided, creates dynamic spaced steps with growing intervals.
        
        Returns:
        - dense_past_inputs (np.ndarray): Dense lookback features. Shape: [samples, dense_lookback, features].
        - spaced_past_inputs (np.ndarray): Spaced lookback features. Shape: [samples, spaced_steps, features] or None.
        - future_inputs (np.ndarray): Future features for the decoder. Shape: [samples, forecast_horizon, future_features].
        - outputs (np.ndarray): Forecast horizon target values. Shape: [samples, forecast_horizon, targets].
        """
        dense_past_inputs, spaced_past_inputs, future_inputs, outputs = [], [], [], []

        # Generate spaced steps
        if spaced_lookback > 0:
            if step_growth_factor is not None:
                # Dynamic step generation
                spaced_steps = [0]  # Start at timestep t
                step_size = 24  # Start with 1 day (24 timesteps)
                total_steps = 0

                while total_steps + step_size < spaced_lookback:
                    total_steps += step_size
                    spaced_steps.append(total_steps)
                    step_size = int(step_size * step_growth_factor)  # Increase step size

                spaced_steps = np.array(spaced_steps)
            else:
                # Fixed step size
                spaced_steps = np.arange(0, spaced_lookback, step=24)
        else:
            spaced_steps = None

        for i in range(len(input_features) - dense_lookback - (max(spaced_steps) if spaced_steps is not None else 0) - forecast_horizon):

            # Extract dense past features
            dense_past_features = input_features[i : i + dense_lookback]
            # Extract spaced past features if applicable
            if spaced_steps is not None:
                spaced_past_indices = i + dense_lookback - spaced_steps[::-1]  # Reverse for chronological order
                spaced_past_features = input_features[spaced_past_indices]
            else:
                spaced_past_features = None

            # Extract target values for the forecast horizon
            target_slice = target[i + dense_lookback : i + dense_lookback + forecast_horizon]
            if len(target_slice) != forecast_horizon:
                continue  # Skip incomplete slices

            # Extract future features
            if future_indices is not None:
                future_features = input_features[i + dense_lookback : i + dense_lookback + forecast_horizon, future_indices]
            else:
                future_features = np.zeros((forecast_horizon, input_features.shape[1]))  # Placeholder if no future features

            # Append to the lists
            dense_past_inputs.append(dense_past_features)
            if spaced_past_features is not None:
                spaced_past_inputs.append(spaced_past_features)
            future_inputs.append(future_features)
            outputs.append(target_slice)

        # Convert to NumPy arrays
        dense_past_inputs = np.array(dense_past_inputs)
        spaced_past_inputs = np.array(spaced_past_inputs) if spaced_steps is not None else None
        future_inputs = np.array(future_inputs)
        outputs = np.array(outputs)

        return dense_past_inputs, spaced_past_inputs, future_inputs, outputs


    def create_datasets(
        features,
        pca_inputs,
        target,
        dense_lookback,
        spaced_lookback,
        forecast_horizon,
        future_cols=None,
        p_train=0.7,
        p_val=0.2,
        p_test=0.1,
        spaced=True,
        step_growth_factor=None
    ):
        """
        Create datasets with dense and optionally spaced lookback for LSTM training.
        """
        assert len(features) == len(target), "Features and target must have the same length"

        hours = len(features)
        future_indices = future_cols if future_cols else None

        # Set dataset sizes
        usable_hours = hours - max(dense_lookback, spaced_lookback if spaced else 0) - forecast_horizon
        print("usable_hours:", usable_hours)
        num_train = int(usable_hours * p_train)
        num_val = int(usable_hours * p_val)
        num_test = usable_hours - num_train - num_val
        pca_inputs = pca_inputs[:usable_hours]

        if spaced_lookback > len(features) - dense_lookback - forecast_horizon:
            spaced_lookback = len(features) - dense_lookback - forecast_horizon
        # Generate features and labels
        if spaced:
            dense_past, spaced_past, future_inputs, outputs = create_features(
                features,
                target,
                dense_lookback,
                spaced_lookback,
                forecast_horizon,
                future_indices,
                step_growth_factor=step_growth_factor,
            )
        else:
            spaced_past = None
            dense_past, _, future_inputs, outputs = create_features(
                features,
                target,
                dense_lookback,
                0,  # No spaced lookback
                forecast_horizon,
                future_indices,
            )

        # Split datasets
        train_dense = dense_past[:num_train]
        val_dense = dense_past[num_train : num_train + num_val]
        test_dense = dense_past[num_train + num_val :]

        if spaced:
            train_spaced = spaced_past[:num_train]
            val_spaced = spaced_past[num_train : num_train + num_val]
            test_spaced = spaced_past[num_train + num_val :]
        else:
            train_spaced = val_spaced = test_spaced = None

        train_future = future_inputs[:num_train]
        val_future = future_inputs[num_train : num_train + num_val]
        test_future = future_inputs[num_train + num_val :]

        train_targets = outputs[:num_train]
        val_targets = outputs[num_train : num_train + num_val]
        test_targets = outputs[num_train + num_val :]

        train_pca = pca_inputs[:num_train]
        val_pca = pca_inputs[num_train : num_train + num_val]
        test_pca = pca_inputs[num_train + num_val :]
        return (
            train_dense,
            train_spaced,
            train_future,
            train_targets,
            val_dense,
            val_spaced,
            val_future,
            val_targets,
            test_dense,
            test_spaced,
            test_future,
            test_targets,
            train_pca,
            val_pca,
            test_pca,
        )


    def perform_pca(dataframe, target_col=None, exclude_cols=None, exclude_scaler=None, variance_threshold=0.01):

        # Exclude target and other specified columns
        exclude_cols = exclude_cols or []
        if target_col:
            exclude_cols.append(target_col)
        feature_cols = [col for col in dataframe.columns if col not in exclude_cols]
        
        # Convert to NumPy array
        feature_data = dataframe[feature_cols].to_numpy()
        not_scaled = dataframe[exclude_scaler].to_numpy()
        # Scale data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(feature_data)
        pca_features = np.concatenate((scaled_data, not_scaled), axis=1)
        # Perform PCA
        pca_model = PCA()
        pca_data = pca_model.fit_transform(pca_features)

        # Explained variance
        explained_variance_ratio = pca_model.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)

        # Find number of components above the variance threshold
        components_above_threshold = np.where(explained_variance_ratio >= variance_threshold)[0]
        optimal_components = len(components_above_threshold)

        # Output results
        return {
            "pca_model": pca_model,
            "scaled_data": scaled_data,
            "pca_data": pca_data,
            "explained_variance_ratio": explained_variance_ratio,
            "cumulative_variance": cumulative_variance,
            "optimal_components": optimal_components,
        }


    def prepare_data(
        file_path,
        dense_lookback,
        spaced_lookback,
        forecast_horizon,
        future_cols,
        target_col,
        spaced=True,
        step_growth_factor=None,
        ):
        """
        Prepare data for forked training with dense and spaced lookback.
        Scales all features except sine/cosine temporal columns and explicitly dropped columns.
        Args:
            file_path (str): Path to the CSV file containing the dataset.
            dense_lookback (int): Number of consecutive timesteps for dense lookback.
            spaced_lookback (int): Maximum lookback period for spaced lookback.
            forecast_horizon (int): Number of timesteps to forecast.
            future_cols (list): Names of future columns.
            target_col (str): Name of the target column.
            spaced (bool): Whether to include spaced lookback features.
            step_growth_factor (float): Factor for dynamic spacing of spaced lookback.
        Returns:
            Scaled datasets (train, val, test) and scalers.
        """
        # Load and preprocess data
        data = pd.read_csv(file_path)
        data = data.drop(columns=["Year", "Month", "Day", "Hour", "Volume_MWh"])

        potential_targets = ["DA", "ID", "Diff"]
        if target_col not in potential_targets:
            raise ValueError(f"Invalid target column: {target_col}. Must be one of {potential_targets}")
        # Drop rest of the potential targets
        potential_targets.remove(target_col)
        data = data.drop(columns=potential_targets)
        # Drop only all-NaN rows
        data = data.dropna(how="all")
        data.ffill(inplace=True)
        data.bfill(inplace=True)

        # Identify columns to exclude from scaling
        exclude_from_scaling = ["Year_Scaled", "Hour_Sin", "Hour_Cos", "Day_Sin", "Day_Cos", "Month_Sin", "Month_Cos"]
        # Separate features and target
        output_data = np.array(data.loc[:, target_col]).reshape(-1, 1)
        future_indices = [data.columns.get_loc(col) for col in future_cols]
        feature_cols = [col for col in data.columns if col not in exclude_from_scaling and col != target_col]
        pca_results = perform_pca(dataframe=data, target_col=target_col, exclude_cols=["ID"], exclude_scaler=exclude_from_scaling, variance_threshold=0.01)
        pca_data = pca_results["pca_data"]
        optimal_components = pca_results["optimal_components"]
        # Select only the optimal components
        pca_features = pca_data[:, :optimal_components]
        # Prepare datasets
        (
            train_dense_past,
            train_spaced_past,
            train_future,
            train_targets,
            val_dense_past,
            val_spaced_past,
            val_future,
            val_targets,
            test_dense_past,
            test_spaced_past,
            test_future,
            test_targets,
            train_pca,
            val_pca,
            test_pca,
        ) = create_datasets(
            features=data.to_numpy(),
            pca_inputs=pca_features,
            target=output_data,
            dense_lookback=dense_lookback,
            spaced_lookback=spaced_lookback,
            forecast_horizon=forecast_horizon,
            future_cols=future_indices,
            p_train=0.7,
            p_val=0.15,
            p_test=0.15,
            spaced=spaced,
            step_growth_factor=step_growth_factor,
        )

        # Initialize scalers for features
        scalers = {col: MinMaxScaler(feature_range=(0, 1)) for col in feature_cols}

        # Apply scaling
        for col in feature_cols:
            col_idx = data.columns.get_loc(col)

            # Fit scaler on training dense past data
            flat_train_dense = train_dense_past[:, :, col_idx].flatten().reshape(-1, 1)
            scalers[col].fit(flat_train_dense)

            # Transform dense past data
            for dataset in [train_dense_past, val_dense_past, test_dense_past]:
                dataset[:, :, col_idx] = scalers[col].transform(
                    dataset[:, :, col_idx].reshape(-1, 1)
                ).reshape(dataset.shape[0], dataset.shape[1])

            # If spaced lookback exists, transform spaced past data
            if spaced:
                # Concatenate dense past and spaced lookback training data
                combined_train_data = np.concatenate([
                    train_dense_past[:, :, col_idx].flatten(),
                    train_spaced_past[:, :, col_idx].flatten()
                ]).reshape(-1, 1)

                # Fit scaler on the combined training data
                scalers[col].fit(combined_train_data)
                for dataset in [train_spaced_past, val_spaced_past, test_spaced_past]:
                    dataset[:, :, col_idx] = scalers[col].transform(
                        dataset[:, :, col_idx].reshape(-1, 1)
                    ).reshape(dataset.shape[0], dataset.shape[1])

        # Scale the target column
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        train_targets = target_scaler.fit_transform(train_targets.reshape(-1, 1)).reshape(train_targets.shape)
        val_targets = target_scaler.transform(val_targets.reshape(-1, 1)).reshape(val_targets.shape)
        test_targets = target_scaler.transform(test_targets.reshape(-1, 1)).reshape(test_targets.shape)

        # Convert datasets to PyTorch tensors
        train_dense_past = torch.tensor(train_dense_past, dtype=torch.float32)
        val_dense_past = torch.tensor(val_dense_past, dtype=torch.float32)
        test_dense_past = torch.tensor(test_dense_past, dtype=torch.float32)
        train_future = torch.tensor(train_future, dtype=torch.float32)
        val_future = torch.tensor(val_future, dtype=torch.float32)
        test_future = torch.tensor(test_future, dtype=torch.float32)
        train_targets = torch.tensor(train_targets, dtype=torch.float32)
        val_targets = torch.tensor(val_targets, dtype=torch.float32)
        test_targets = torch.tensor(test_targets, dtype=torch.float32)
        train_pca = torch.tensor(train_pca, dtype=torch.float32)
        val_pca = torch.tensor(val_pca, dtype=torch.float32)
        test_pca = torch.tensor(test_pca, dtype=torch.float32)

        if spaced:
            train_spaced_past = torch.tensor(train_spaced_past, dtype=torch.float32)
            val_spaced_past = torch.tensor(val_spaced_past, dtype=torch.float32)
            test_spaced_past = torch.tensor(test_spaced_past, dtype=torch.float32)
            return (
                train_dense_past,
                train_spaced_past,
                train_future,
                train_pca,
                train_targets,
                val_dense_past,
                val_spaced_past,
                val_future,
                val_pca,
                val_targets,
                test_dense_past,
                test_spaced_past,
                test_future,
                test_pca,
                test_targets,
                target_scaler,
            )
        else:
            return (
                train_dense_past,
                train_future,
                train_pca,
                train_targets,
                val_dense_past,
                val_future,
                val_pca,
                val_targets,
                test_dense_past,
                test_future,
                test_pca,
                test_targets,
                target_scaler,
            )