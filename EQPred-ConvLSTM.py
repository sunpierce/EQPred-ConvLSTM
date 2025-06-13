import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import (Model, Sequential)
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, ConvLSTM2D, Concatenate,
    Flatten, Dense, TimeDistributed, Dropout, LSTM,
    BatchNormalization, AveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (confusion_matrix, roc_curve)
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime, timedelta
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

# Verify GPU availability
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print("TensorFlow version:", tf.__version__)

# Configuration
IMAGE_DIR = "/content/drive/MyDrive/groundwater_maps_global_percentile_scaled"
LABEL_CSV = "/content/drive/MyDrive/Earthquake_Occurrence.csv"
IMG_SIZE = (128, 128)
SEQ_LENGTH = 7
BATCH_SIZE = 100
PATIENCE = 10
EPOCHS = 100
LAMBDA = 0.6
LEARN_RATE = 1e-7
THRESHOLD_DIFF = 1e-5
CHANNEL = 16

# Cache configuration
CACHE_DIR = "/content/drive/MyDrive/eqpred_cache"
DATA_CACHE_FILE = os.path.join(CACHE_DIR, "preprocessed_data.npz")
SPLITS_CACHE_FILE = os.path.join(CACHE_DIR, "temporal_splits.pkl")

def ensure_cache_dir():
    """Ensure cache directory exists"""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        print(f"Created cache directory at {CACHE_DIR}")
    else:
        print(f"Cache directory exists at {CACHE_DIR}")

def load_and_preprocess_data():
    """Load earthquake labels and image sequences with normalization (0-1).
    Returns:
        X: Array of image sequences (n_samples, seq_len, height, width, 1)
        y: Array of earthquake labels (1 if earthquake occurred next day)
        sample_dates: List of dates corresponding to each sample (the date we're checking for an earthquake)
    """
    # Load earthquake labels
    labels_df = pd.read_csv(LABEL_CSV)
    labels_df['date'] = pd.to_datetime(labels_df[['Year', 'Month', 'Day']])
    labels_df.set_index('date', inplace=True) # Set 'date' as the index for easy lookup

    # Create date range for image sequences
    date_range = pd.date_range(start='2009-04-01', end='2023-03-31')

    # Create dataset
    X, y, sample_dates = [], [], [] # X = images, y = labels, sample_dates = dates
    valid_dates_count = 0 # Tracks how many images were successfully loaded

    for i in range(SEQ_LENGTH, len(date_range)):
        current_date = date_range[i]
        if current_date in labels_df.index:
            # Load image sequence (t-29 to t)
            seq = []
            missing_images = False

            for j in range(i - SEQ_LENGTH, i): # Loop over SEQ_LENGTH days before current_date
                img_date = date_range[j]
                img_path = os.path.join(IMAGE_DIR, f"{img_date.strftime('%Y-%m-%d')}.png")

                if os.path.exists(img_path):
                    try:
                        img = np.array(Image.open(img_path).convert('L')) # Load as grayscale (0-255)
                        img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
                        seq.append(img) # Add to sequence
                        valid_dates_count += 1
                    except Exception as e:
                        print(f"Error loading {img_path}: {str(e)}")
                        missing_images = True
                        break
                else:
                    missing_images = True
                    break

            if not missing_images and len(seq) == SEQ_LENGTH:
                # Get all label values for the current date
                label_values = labels_df.loc[current_date, 'Occur']
                if isinstance(label_values, (pd.Series, np.ndarray)):
                    # If multiple labels exist for this date, duplicate the sequence for each label
                    if len(label_values) > 1:
                        print(f"Multiple labels found for date {current_date}. Duplicate the sequence...")
                    for label in label_values:
                        y.append(label)  # Append each label
                        X.append(np.array(seq))  # Add the same sequence multiple times
                        sample_dates.append(current_date)  # Add the same date multiple times
                else:
                    # Single label case (scalar)
                    y.append(label_values)
                    X.append(np.array(seq))
                    sample_dates.append(current_date)

    # Convert y to numpy array before calculating stats
    y_array = np.array(y)
    positive_count = np.sum(y_array)
    total_samples = len(y_array)

    print(f"Loaded {total_samples} valid sequences ({valid_dates_count} total images)")
    print(f"Positive samples: {positive_count}/{total_samples} ({positive_count/total_samples:.2%})")

    return np.array(X)[..., np.newaxis], y_array, sample_dates  # Add channel dimension

# Model Architecture
def create_model():
    input_layer = Input(shape=(SEQ_LENGTH, IMG_SIZE[0], IMG_SIZE[1], 1))

    # Block 1: Artificial Channel Creation
    x = TimeDistributed(Conv2D(CHANNEL, (3, 3), padding='same',
                               kernel_regularizer=regularizers.l2(1e-4)))(input_layer)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = TimeDistributed(MaxPooling2D((3, 3), strides=2, padding='valid'))(x)
    # x = TimeDistributed(AveragePooling2D((3, 3), strides=2, padding='valid'))(x)

    # Block 2: Spatiotemporal Feature Extraction
    # Branch 1 (3x3 kernel)
    # b1 = TimeDistributed(Conv2D(32, (1, 1), padding='same', activation='relu'))(x)
    b1 = ConvLSTM2D(CHANNEL, (3, 3), padding='same', return_sequences=True, activation='tanh',
                    kernel_regularizer=regularizers.l2(1e-4),
                    recurrent_regularizer=regularizers.l2(1e-4),
                    bias_regularizer=regularizers.l2(1e-4))(x)
    b1 = ConvLSTM2D(CHANNEL, (3, 3), padding='same', return_sequences=True, activation='tanh',
                    kernel_regularizer=regularizers.l2(1e-4),
                    recurrent_regularizer=regularizers.l2(1e-4),
                    bias_regularizer=regularizers.l2(1e-4))(b1)
    b1 = ConvLSTM2D(CHANNEL, (3, 3), padding='same', return_sequences=True, activation='tanh',
                    kernel_regularizer=regularizers.l2(1e-4),
                    recurrent_regularizer=regularizers.l2(1e-4),
                    bias_regularizer=regularizers.l2(1e-4))(b1)
    b1 = ConvLSTM2D(CHANNEL, (3, 3), padding='same', return_sequences=False, activation='tanh',
                    kernel_regularizer=regularizers.l2(1e-4),
                    recurrent_regularizer=regularizers.l2(1e-4),
                    bias_regularizer=regularizers.l2(1e-4))(b1)

    # Branch 2 (5x5 kernel)
    # b2 = TimeDistributed(Conv2D(32, (1, 1), padding='same', activation='relu'))(x)
    b2 = ConvLSTM2D(CHANNEL, (5, 5), padding='same', return_sequences=True, activation='tanh',
                    kernel_regularizer=regularizers.l2(1e-4),
                    recurrent_regularizer=regularizers.l2(1e-4),
                    bias_regularizer=regularizers.l2(1e-4))(x)
    b2 = ConvLSTM2D(CHANNEL, (5, 5), padding='same', return_sequences=True, activation='tanh',
                    kernel_regularizer=regularizers.l2(1e-4),
                    recurrent_regularizer=regularizers.l2(1e-4),
                    bias_regularizer=regularizers.l2(1e-4))(b2)
    b2 = ConvLSTM2D(CHANNEL, (5, 5), padding='same', return_sequences=True, activation='tanh',
                    kernel_regularizer=regularizers.l2(1e-4),
                    recurrent_regularizer=regularizers.l2(1e-4),
                    bias_regularizer=regularizers.l2(1e-4))(b2)
    b2 = ConvLSTM2D(CHANNEL, (5, 5), padding='same', return_sequences=False, activation='tanh',
                    kernel_regularizer=regularizers.l2(1e-4),
                    recurrent_regularizer=regularizers.l2(1e-4),
                    bias_regularizer=regularizers.l2(1e-4))(b2)

    # Concatenate branches
    concat = Concatenate()([b1, b2])

    # Block 3: Forecasting Network
    # x = MaxPooling2D((3, 3), strides=2, padding='valid')(concat)
    x = AveragePooling2D((3, 3), strides=2, padding='valid')(concat)
    x = Flatten()(x)
    x = Dense(16, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(16, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(16, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(16, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(16, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(16, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Dropout(0.3)(x)

    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output)
    model.summary()
    return model

# Simplified model architecture (baseline)
def build_simpler_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        TimeDistributed(Conv2D(CHANNEL, (3, 3), padding='same',
                               kernel_regularizer=regularizers.l2(1e-4))),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        TimeDistributed(MaxPooling2D()),
        TimeDistributed(Conv2D(CHANNEL, (3, 3), padding='same',
                               kernel_regularizer=regularizers.l2(1e-4))),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        TimeDistributed(MaxPooling2D()),
        TimeDistributed(Flatten()),
        LSTM(CHANNEL, return_sequences=True,
             kernel_regularizer=regularizers.l2(1e-4),
             recurrent_regularizer=regularizers.l2(1e-4),
             bias_regularizer=regularizers.l2(1e-4)),
        LSTM(CHANNEL, return_sequences=True,
             kernel_regularizer=regularizers.l2(1e-4),
             recurrent_regularizer=regularizers.l2(1e-4),
             bias_regularizer=regularizers.l2(1e-4)),
        LSTM(CHANNEL, return_sequences=True,
             kernel_regularizer=regularizers.l2(1e-4),
             recurrent_regularizer=regularizers.l2(1e-4),
             bias_regularizer=regularizers.l2(1e-4)),
        LSTM(CHANNEL, return_sequences=False,
             kernel_regularizer=regularizers.l2(1e-4),
             recurrent_regularizer=regularizers.l2(1e-4),
             bias_regularizer=regularizers.l2(1e-4)),
        Dense(16, kernel_regularizer=regularizers.l2(1e-4)),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        Dropout(0.3),
        Dense(16, kernel_regularizer=regularizers.l2(1e-4)),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        Dropout(0.3),
        Dense(16, kernel_regularizer=regularizers.l2(1e-4)),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        Dropout(0.3),
        Dense(16, kernel_regularizer=regularizers.l2(1e-4)),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.summary()
    return model

# Custom Metrics and Threshold Optimization
def balanced_score(y_true, y_pred, threshold, lambda_param=LAMBDA):
    y_pred_bin = (y_pred >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    score = lambda_param * fnr + (1 - lambda_param) * fpr
    return score, fpr, fnr

# Find optimal threshold using balanced score only
def find_optimal_threshold(y_true, y_pred_prob):
    best_threshold = 0.5
    best_score = float('inf')
    for threshold in np.arange(0, 1, THRESHOLD_DIFF):
        if threshold == 0.1: print("Optimizing Threshold = 0.1 ...")
        if threshold == 0.5: print("Optimizing Threshold = 0.5 ...")
        if threshold == 0.9: print("Optimizing Threshold = 0.9 ...")
        score, _, _ = balanced_score(y_true, y_pred_prob, threshold)
        if score < best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold, best_score

# Include other metrics
def find_optimal(y_true, y_pred_proba, method):
    best_threshold = 0.5
    best_score = float('inf')
    # Iterate over candidate thresholds
    for threshold in np.arange(0, 1, THRESHOLD_DIFF):
        if threshold == 0.1: print("Optimizing Threshold = 0.1 ...")
        if threshold == 0.5: print("Optimizing Threshold = 0.5 ...")
        if threshold == 0.9: print("Optimizing Threshold = 0.9 ...")

        y_pred = (y_pred_proba >= threshold).astype(int)
        # Compute confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        # Calculate FPR and FNR
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        # Compute the objective
        if method == 'sum':
            score = fpr + fnr
        elif method == 'euclidean':
            score = np.sqrt(fpr**2 + fnr**2)
        elif method == 'minimax':
            score = max(fpr, fnr)
        elif method == 'balance':
            score, _, _ = balanced_score(y_true, y_pred_proba, threshold)
        else:
            raise ValueError("Method must be 'sum', 'euclidean', 'minimax', or 'balance'.")

        # Update best threshold if score improves
        if score < best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score

# Temporal Splitting
def temporal_split(dates):
    """Produces exactly 3 splits with 3-month test periods ending on month boundaries"""
    # Convert to pandas Series for date-based indexing
    y_series = pd.Series(range(len(dates)), index=dates)

    # Define fixed 3-month splits (test periods end on last day of month)
    split_points = [
        # Split 1: Test = July-Sep (92 days)
        {
            'val_start': datetime(2022, 4, 1),
            'val_end': datetime(2022, 6, 30),
            'test_start': datetime(2022, 7, 1),
            'test_end': datetime(2022, 9, 30)
        },
        # Split 2: Test = Oct-Dec (92 days)
        {
            'val_start': datetime(2022, 7, 1),
            'val_end': datetime(2022, 9, 30),
            'test_start': datetime(2022, 10, 1),
            'test_end': datetime(2022, 12, 31)
        },
        # Split 3: Test = Jan-Mar (90 days)
        {
            'val_start': datetime(2022, 10, 1),
            'val_end': datetime(2022, 12, 31),
            'test_start': datetime(2023, 1, 1),
            'test_end': datetime(2023, 3, 31)
        }
    ]

    splits = []
    for split in split_points:
        # Get indices for each split
        train_idx = y_series[y_series.index < split['val_start']].values
        val_idx = y_series[
            (y_series.index >= split['val_start']) &
            (y_series.index <= split['val_end'])
        ].values
        test_idx = y_series[
            (y_series.index >= split['test_start']) &
            (y_series.index <= split['test_end'])
        ].values

        if len(train_idx) > 0 and len(val_idx) > 0 and len(test_idx) > 0:
            splits.append({
                'train_idx': train_idx,
                'val_idx': val_idx,
                'test_idx': test_idx,
                'val_start': split['val_start'].strftime('%Y-%m-%d'),
                'val_end': split['val_end'].strftime('%Y-%m-%d'),
                'test_start': split['test_start'].strftime('%Y-%m-%d'),
                'test_end': split['test_end'].strftime('%Y-%m-%d')
            })

    # Strict validation
    if len(splits) != 3:
        raise ValueError(f"Expected 3 splits but got {len(splits)}. Check date ranges.")

    return splits

# Plotting function
def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Weighted Cross Entropy Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

def save_preprocessed_data(X, y, dates, filepath):
    """Safely save preprocessed data with error handling"""
    try:
        np.savez(
            filepath,
            X=X,
            y=y,
            dates=np.array([d.strftime('%Y-%m-%d') for d in dates])
        )
        print(f"Successfully saved preprocessed data to {filepath}")
        return True
    except Exception as e:
        print(f"Error saving preprocessed data: {str(e)}")
        return False

def load_preprocessed_data(filepath):
    """Safely load preprocessed data with error handling"""
    try:
        with np.load(filepath) as data:
            X = data['X']
            y = data['y']
            dates = [datetime.strptime(d, '%Y-%m-%d') for d in data['dates']]
        print(f"Successfully loaded preprocessed data from {filepath}")
        return X, y, dates
    except Exception as e:
        print(f"Error loading preprocessed data: {str(e)}")
        return None, None, None

def get_file_size_gb(filepath):
    """Returns file size in GB rounded to 2 decimals"""
    if os.path.exists(filepath):
        size_bytes = os.path.getsize(filepath)
        size_gb = size_bytes / (1024 ** 3)  # Convert bytes to GB
        return round(size_gb, 2)
    return 0

# Main Execution
def main():
    # Mount Google Drive (ensure this is at the start)
    # from google.colab import drive
    # drive.mount('/content/drive', force_remount=True)

    # Ensure cache directory exists
    ensure_cache_dir()

    # Check cache status
    data_cache_exists = os.path.exists(DATA_CACHE_FILE)
    splits_cache_exists = os.path.exists(SPLITS_CACHE_FILE)

    # Debug: Print cache status
    print(f"\nCache status:")
    if os.path.exists(DATA_CACHE_FILE):
        print(f"- Preprocessed data: Exists ({get_file_size_gb(DATA_CACHE_FILE)} GB)")
    else:
        print(f"- Preprocessed data: Missing")

    if os.path.exists(SPLITS_CACHE_FILE):
        print(f"- Temporal splits: Exists ({get_file_size_gb(SPLITS_CACHE_FILE)} GB)")
    else:
        print(f"- Temporal splits: Missing")

    # Load or process data
    if data_cache_exists and splits_cache_exists:
        print("\nAttempting to load from cache...")
        X, y, dates = load_preprocessed_data(DATA_CACHE_FILE)

        if X is not None:
            with open(SPLITS_CACHE_FILE, 'rb') as f:
                splits = pickle.load(f)
            print(f"Loaded {len(X)} samples and {len(splits)} splits from cache")
        else:
            print("Cache loading failed, reprocessing data...")
            X, y, dates = load_and_preprocess_data()
            splits = temporal_split(dates)

            # Retry saving cache
            if not save_preprocessed_data(X, y, dates, DATA_CACHE_FILE):
                print("Warning: Failed to save preprocessed data cache")
            with open(SPLITS_CACHE_FILE, 'wb') as f:
                pickle.dump(splits, f)
    else:
        print("\nCache not complete, processing data...")
        X, y, dates = load_and_preprocess_data()
        splits = temporal_split(dates)

        # Save both cache files
        cache_success = save_preprocessed_data(X, y, dates, DATA_CACHE_FILE)
        if cache_success:
            print(f"Saved preprocessed data ({get_file_size_gb(DATA_CACHE_FILE)} GB)")
        if not cache_success:
            print(f"Warning: Failed to save preprocessed data cache ({get_file_size_gb(DATA_CACHE_FILE)} GB)")

        with open(SPLITS_CACHE_FILE, 'wb') as f:
            pickle.dump(splits, f)
        print(f"Saved temporal splits ({get_file_size_gb(SPLITS_CACHE_FILE)} GB)")

    # Verify data shapes
    print("\nData verification:")
    print(f"- X shape: {X.shape if X is not None else 'None'}")
    print(f"- y shape: {y.shape if y is not None else 'None'}")
    print(f"- Dates length: {len(dates) if dates is not None else 'None'}")
    print(f"- Number of splits: {len(splits) if splits is not None else 'None'}")

    # Training and evaluation
    results = []
    for i, split in enumerate(splits):
        print(f"\n===================== Training on Split {i+1}/{len(splits)} =====================")
        print(f"Validation Period: {split['val_start']} to {split['val_end']}")
        print(f"Test Period: {split['test_start']} to {split['test_end']}")

        X_train, y_train = X[split['train_idx']], y[split['train_idx']]
        X_val, y_val = X[split['val_idx']], y[split['val_idx']]
        X_test, y_test = X[split['test_idx']], y[split['test_idx']]

        print(f"Train: {len(X_train)} samples ({y_train.mean():.4f} positive)")
        print(f"Val: {len(X_val)} samples ({y_val.mean():.4f} positive)")
        print(f"Test: {len(X_test)} samples ({y_test.mean():.4f} positive)")

        # Compute class weights for weighted binary cross-entropy
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))
        print(f"Computed class weights: {class_weights}")

        # Implement EQPred-ConvLSTM
        # model = create_model()

        # Implement CNN+LSTM
        model = build_simpler_model(input_shape=(SEQ_LENGTH, IMG_SIZE[0], IMG_SIZE[1], 1))

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=LEARN_RATE),
            loss='binary_crossentropy'
        )

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001
        )

        print("Starting training...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=[early_stop],
            class_weight=class_weights,
            verbose=1
        )

        # Plot training history
        plot_history(history)

        # Threshold optimization
        print("Optimizing decision threshold...")
        y_pred = model.predict(X_val, verbose=0).flatten()
        opt_threshold, val_score = find_optimal_threshold(y_val, y_pred)
        print(f"Optimal threshold: {opt_threshold:.5f}, Validation Balanced Score: {val_score:.4f}")

        # Final evaluation
        print("Evaluating on test set...")
        y_pred = model.predict(X_test, verbose=0).flatten()
        test_score, fpr, fnr = balanced_score(y_test, y_pred, opt_threshold, LAMBDA)
        # Print all predicted probabilities
        for i, prob in enumerate(y_pred):
            print(f"Instance {i}: Predicted Probability = {prob:.6f}")

        # Confusion matrix
        y_pred_bin = (y_pred >= opt_threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred_bin)
        tn, fp, fn, tp = cm.ravel()

        results.append({
            'split': i+1,
            'val_period': f"{split['val_start']} to {split['val_end']}",
            'test_period': f"{split['test_start']} to {split['test_end']}",
            'threshold': opt_threshold,
            'FPR': fpr,
            'FNR': fnr,
            'BalancedScore': test_score,
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn,
            'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0
        })

        print(f"Test Results - FPR: {fpr:.4f}, FNR: {fnr:.4f}, BalancedScore: {test_score:.4f}")
        print(f"Confusion Matrix:\n{cm}")

    # Save and print final results
    results_df = pd.DataFrame(results)
    print("\n=============== Final Results ===============")
    print(results_df)

    # Save results to CSV
    results_csv = os.path.join(CACHE_DIR, "results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"Results saved to {results_csv}")

if __name__ == "__main__":
    main()