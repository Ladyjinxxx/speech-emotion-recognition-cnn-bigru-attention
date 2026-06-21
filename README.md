# Hybrid CNN-BiGRU with Self-Attention for Speech Emotion Recognition Using MFCC and Mel Spectrogram Features

A deep learning model for classifying human emotion from speech audio, built for **DS 175 – Deep Neural Networks** (BS Data Science).

**Authors:** Geverola, Lady Jean Y. · Valdez, Adrienne Jill B.

---

## Description

This project implements a hybrid deep learning architecture for **Speech Emotion Recognition (SER)**. The model combines:

- **CNN layers** — extract local acoustic and spectral patterns from speech features
- **Bidirectional GRU (BiGRU)** — capture temporal dependencies across the speech sequence (past + future context)
- **Self-Attention mechanism** — lets the model focus on the most emotionally significant segments of speech

Input features are **MFCC** and **Mel Spectrogram** representations extracted from the **RAVDESS** dataset. Data augmentation (noise injection, pitch shifting, time stretching) is applied to improve generalization.

## Problem Statement

Recognizing emotion from speech is difficult because emotional expression varies widely in tone, pitch, speaking style, intensity, and pronunciation. This project develops a hybrid CNN-BiGRU + Self-Attention model to classify speech audio into one of **8 emotion classes**:

`Neutral` · `Calm` · `Happy` · `Sad` · `Angry` · `Fearful` · `Disgust` · `Surprised`

SER has real-world relevance in virtual assistants, customer service systems, healthcare monitoring, intelligent tutoring, and human-computer interaction.

## Dataset

- **[RAVDESS — Ryerson Audio-Visual Database of Emotional Speech and Song](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)**
- Downloaded via `kagglehub`
- 8 emotion classes, mapped from RAVDESS filename codes (`01`–`08`)

## Methodology

### 1. Preprocessing & Augmentation
- Random noise injection
- Pitch shifting
- Time stretching

### 2. Feature Extraction
- **MFCC** (40 coefficients) + **Mel Spectrogram** (128 filters) → combined into a 168-dimensional feature vector per time step
- Sequences padded/truncated to a fixed length of **174 time steps**
- Final feature shape per sample: `(174, 168)`
- Features normalized to roughly the `[-1, 1]` range

### 3. Label Encoding
- Emotion labels → integer-encoded → one-hot encoded for multiclass classification

### 4. Train/Test Split
- 80% train / 20% test, stratified by class, `random_state=42`
- Training set: 9,216 samples · Testing set: 2,304 samples

### 5. Model Architecture
```
Input (174, 168)
   ↓
CNN Block 1: Conv1D(64) → BatchNorm → MaxPool → Dropout(0.3)
   ↓
CNN Block 2: Conv1D → BatchNorm → MaxPool → Dropout
   ↓
Bidirectional GRU (128 units, return_sequences=True)
   ↓
Self-Attention (custom layer)
   ↓
Dense (128, ReLU) → Dropout
   ↓
Dense (8, Softmax) → predicted emotion
```

### 6. Training
- Optimizer: **Adam**
- Regularization: Dropout, Batch Normalization, **EarlyStopping** (monitors validation performance)

## Results

| Metric | Score |
|---|---|
| **Test Accuracy** | ~94.10% |
| **Test Loss** | ~0.1654 |

Evaluation included a full classification report (precision/recall/F1 per class), confusion matrix, and sample-level prediction inspection.

**Common confusion pairs** (acoustically similar emotions):
- Calm ↔ Neutral
- Fearful ↔ Surprised
- Sad ↔ Calm

## Tech Stack

| Purpose | Library |
|---|---|
| Deep learning | TensorFlow / Keras |
| Audio processing | Librosa |
| ML utilities (split, encoding, metrics) | Scikit-learn |
| Data handling | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Dataset download | KaggleHub |
| Environment | Google Colab |

## Requirements

```bash
pip install numpy pandas librosa matplotlib seaborn kagglehub tensorflow scikit-learn
```

## Usage

1. Clone this repository
2. Open `DS175_FinalProject_Geverola_Valdez.ipynb` in Jupyter or Google Colab
3. Run all cells in order — the notebook downloads the RAVDESS dataset automatically via `kagglehub`
4. Trained model evaluates on the held-out test set with accuracy/loss, confusion matrix, and classification report printed at the end

## Possible Improvements

- Multi-Head / Transformer-based attention instead of single self-attention
- Additional features: Chroma, Spectral Contrast, Tonnetz
- Hyperparameter tuning (learning rate, batch size, GRU units, dropout, filters)
- Training on larger, multilingual, real-world noisy datasets
- Real-time deployment for mobile / virtual assistant use cases

## Ethical Considerations

Speech carries sensitive personal information, so SER systems must handle voice data with proper consent and secure storage. Emotion datasets may also underrepresent certain accents, languages, and demographics, which can introduce bias. SER predictions should support — not replace — human judgment in sensitive applications.

## References

**Papers**
1. Vaswani et al. (2017). *Attention Is All You Need*. NeurIPS. [arxiv.org/pdf/1706.03762](https://arxiv.org/pdf/1706.03762)
2. Cho et al. (2014). *Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation*. EMNLP. [arxiv.org/pdf/1406.1078](https://arxiv.org/pdf/1406.1078)
3. Hochreiter & Schmidhuber (1997). *Long Short-Term Memory*. Neural Computation.
4. Livingstone & Russo (2018). *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)*. PLoS ONE.

**Dataset**
- [RAVDESS Emotional Speech Audio (Kaggle)](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)

**Documentation**
- [TensorFlow/Keras](https://www.tensorflow.org/api_docs) · [Librosa](https://librosa.org/doc/latest/index.html) · [Scikit-learn](https://scikit-learn.org/stable/documentation.html)
