import os
import glob
from pathlib import Path
import random

import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# -----------------------
# USER CONFIG
# -----------------------
DATA_DIR = r"C:\Users\LAVANYA\Downloads\datasetis\audio\audio\44100"   # point this to your 44100 folder
SAMPLE_RATE = 44100
SEGMENT_SECONDS = 5
SEGMENT_SAMPLES = SEGMENT_SECONDS * SAMPLE_RATE  # 5 * 44100 = 220500
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
POWER = 2.0  # power spectrogram (2.0 = energy)
RANDOM_SEED = 42

BATCH_SIZE = 32
EPOCHS = 30
TEST_SPLIT = 0.1
VALIDATION_SPLIT = 0.2  # fraction of remaining after extracting test
MODEL_SAVE_PATH = "audio_44100_cnn.h5"
def list_wav_files(data_dir):
    # return list of wav file paths (recursively)
    pattern = os.path.join(data_dir, "**", "*.wav")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"No .wav files found under {data_dir}")
    return files
def load_wav(path, sr=SAMPLE_RATE):
    audio, file_sr = sf.read(path, always_2d=False)
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    # Resample if needed
    if file_sr != sr:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=file_sr, target_sr=sr)
    # Ensure length exactly SEGMENT_SAMPLES (pad or trim)
    if len(audio) < SEGMENT_SAMPLES:
        pad_len = SEGMENT_SAMPLES - len(audio)
        audio = np.pad(audio, (0, pad_len), mode='constant')
    elif len(audio) > SEGMENT_SAMPLES:
        audio = audio[:SEGMENT_SAMPLES]
    return audio.astype(np.float32)

def compute_mel_db(segment, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=POWER):
    mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=power)
    mel_db = librosa.power_to_db(mel, ref=np.max)  # log scaled
    # normalize to 0..1
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-9)
    return mel_norm.astype(np.float32)  # shape: (n_mels, time_frames)

def extract_label_from_path(path, data_dir):
    """
    Strategy:
     - If file is inside a subfolder of data_dir, use the immediate subfolder name as class.
     - Else, use filename prefix before first '-' (e.g., '1-137-A-32.wav' -> '1').
    Adjust this function if your class encoding differs.
    """
    p = Path(path)
    try:
        rel = p.relative_to(data_dir)
        parts = rel.parts
        if len(parts) >= 2:
            # path like <class_name>/<wav>
            return parts[0]
    except Exception:
        pass
    # fallback: filename prefix before first '-'
    stem = p.stem
    if '-' in stem:
        return stem.split('-', 1)[0]
    # final fallback: parent folder name
    return p.parent.name

# -----------------------
# Build dataset arrays
# -----------------------
def build_dataset_arrays(data_dir):
    wav_files = list_wav_files(data_dir)
    X = []
    y = []
    labels_set = set()
    print(f"Found {len(wav_files)} WAV files. Processing...")

    for p in tqdm(wav_files):
        label = extract_label_from_path(p, data_dir)
        labels_set.add(label)
        audio = load_wav(p, sr=SAMPLE_RATE)
        mel = compute_mel_db(audio)
        # add channel dim for Keras Conv2D: (H, W) -> (H, W, 1)
        mel = np.expand_dims(mel, axis=-1)
        X.append(mel)
        y.append(label)

    X = np.stack(X, axis=0)  # (N, n_mels, time_frames, 1)
    label_list = sorted(list(labels_set))
    label_to_idx = {name: idx for idx, name in enumerate(label_list)}
    y_idx = np.array([label_to_idx[l] for l in y], dtype=np.int32)

    print(f"Built dataset: X.shape={X.shape}, num_classes={len(label_list)}")
    return X, y_idx, label_list

# -----------------------
# Model
# -----------------------
def build_cnn(input_shape, num_classes):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inp, outputs=out)
    return model

# -----------------------
# Training pipeline
# -----------------------
def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    X, y, label_names = build_dataset_arrays(DATA_DIR)

    # Split: test first, then train/val
    X_rest, X_test, y_rest, y_test = train_test_split(X, y, test_size=TEST_SPLIT, stratify=y, random_state=RANDOM_SEED)
    val_frac_of_rest = VALIDATION_SPLIT / (1.0 - TEST_SPLIT)
    X_train, X_val, y_train, y_val = train_test_split(X_rest, y_rest, test_size=val_frac_of_rest, stratify=y_rest, random_state=RANDOM_SEED)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    input_shape = X.shape[1:]  # (n_mels, time_frames, 1)
    num_classes = len(label_names)
    model = build_cnn(input_shape, num_classes)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    cbs = [
        callbacks.ModelCheckpoint("best_model.h5", save_best_only=True, monitor='val_accuracy', mode='max'),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
        callbacks.EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True)
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cbs)

    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")

    model.save(MODEL_SAVE_PATH)
    print(f"Saved model to {MODEL_SAVE_PATH}")

    # Save label mapping
    with open("label_names.txt", "w") as f:
        for name in label_names:
            f.write(name + "\n")
    print("Saved label_names.txt")

if __name__ == "__main__":
    main()
