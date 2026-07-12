# 🔍 PixelTruth — AI-Powered Deepfake Detector

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/Accuracy-95%25-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/GSSoC-2026-purple?style=for-the-badge" />
</p>

> **PixelTruth** is an AI-powered deepfake detection system for social media images. It uses a custom Convolutional Neural Network (CNN) with advanced preprocessing to classify images as **real or AI-generated** with **95% accuracy**.

---
 ## Dataset download link -https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
## 📌 Table of Contents

- [About](#about)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Performance](#performance)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [GSSoC 2026](#gssoc-2026)
- [License](#license)

---

## 🧠 About

With the rise of AI-generated media, detecting deepfakes has become critical for media integrity and combating misinformation. PixelTruth addresses this by providing a fast, accurate, and accessible deepfake detection tool built on deep learning.

It analyzes visual artifacts introduced during image synthesis and classifies images in real-time through an intuitive web dashboard.

---

## ✨ Features

- 🖼️ **Real-time image analysis** — supports JPG, PNG, WebP formats
- 📊 **Confidence scores** — shows probability of real vs. fake
- 🎨 **Modern Streamlit dashboard** — glassmorphism UI with visual feedback
- ⚡ **Fast inference** — lightweight model optimized for speed
- 🔬 **Custom preprocessing pipeline** — enhanced artifact detection

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| TensorFlow / Keras | CNN model training & inference |
| OpenCV | Image preprocessing |
| Streamlit | Web dashboard |
| NumPy / Matplotlib | Data handling & visualization |

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Accuracy | 95% |
| Input Size | 96 x 96 px |
| Supported Formats | JPG, PNG, WebP |

---

## 📐 Model Input & Preprocessing Contract

All models produced by training scripts and executed by inference pipelines comply with the **Canonical Model Version v1.0.0** contract. 

- **Model Version**: `v1.0.0` (MobileNetV2 backbone with embedded rescaling)
- **Raw Preprocessing**: Images are resized to **96x96 pixels** and converted to RGB order, with float32 pixel values kept in the range **`[0.0, 255.0]`**.
- **Internal Model Scaling**: The model embeds a Keras `Rescaling(scale=1./127.5, offset=-1.)` layer as its first preprocessing step. This maps the raw `[0.0, 255.0]` inputs to the range **`[-1.0, 1.0]`** (the canonical input range required by the pretrained MobileNetV2 backbone).
- **Consistency**: Both training-time generators and inference-time pipelines pass raw `[0.0, 255.0]` pixel values directly to the model, ensuring perfect compatibility between training and production environments.

---

## ⚙️ Installation

> **⚠️ Prerequisites:**
> - **Python Version:** This project strictly requires **Python 3.11**.
>   (Newer versions like Python 3.12 are currently incompatible with the required TensorFlow dependencies).

> - **Windows Users:** 
>   - You must have the [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) installed to run TensorFlow.
>   - If the app crashes on launch with a DLL error, install `tensorflow-cpu` instead of standard `tensorflow`.

```bash
# 1. Clone the repository
git clone https://github.com/Piyush-Sharma788/PixelTruth.git
cd PixelTruth

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Model setup

PixelTruth also needs a trained model file before it can make predictions.

#### Option 1: automatic download on first run

Set a direct download URL for the trained model (for example, a GitHub Release asset) and the app will fetch it automatically when it starts:

```bash
# On Linux/macOS
export PIXELTRUTH_MODEL_URL=https://your-release-link/deepfake_detection_model.h5
export PIXELTRUTH_MODEL_SHA256=<optional-sha256>
streamlit run app.py

# On Windows (Command Prompt)
set PIXELTRUTH_MODEL_URL=https://your-release-link/deepfake_detection_model.h5
set PIXELTRUTH_MODEL_SHA256=<optional-sha256>
streamlit run app.py

# On Windows (PowerShell)
$env:PIXELTRUTH_MODEL_URL="https://your-release-link/deepfake_detection_model.h5"
$env:PIXELTRUTH_MODEL_SHA256="<optional-sha256>"
streamlit run app.py
```

#### Option 2: manual download script

Download the model with the helper script:

```bash
python scripts/download_model.py --url https://your-release-link/deepfake_detection_model.h5 --dest deepfake_detection_model.h5
```

#### Option 3: place the file manually

1. Download or generate the trained Keras model.
2. Save it as `deepfake_detection_model.h5` in the project root, or set `PIXELTRUTH_MODEL_PATH` to the file location.
3. Restart the app after placing the model file.

If you keep the model in a different folder, export the path before launching Streamlit:

```bash
# On Linux/macOS
export PIXELTRUTH_MODEL_PATH=/full/path/to/deepfake_detection_model.h5
streamlit run app.py

# On Windows (Command Prompt)
set PIXELTRUTH_MODEL_PATH=C:\full\path\to\deepfake_detection_model.h5
streamlit run app.py

# On Windows (PowerShell)
$env:PIXELTRUTH_MODEL_PATH="C:\full\path\to\deepfake_detection_model.h5"
streamlit run app.py
```

If you receive the model from a release asset or shared download link, copy it into the project root first:

```bash
cp /path/to/downloaded/deepfake_detection_model.h5 ./deepfake_detection_model.h5
```

If you publish the model as a GitHub Release asset, the recommended setup is:

1. Upload `deepfake_detection_model.h5` to the release.
2. Copy the release asset URL into `PIXELTRUTH_MODEL_URL`.
3. Optionally record the SHA256 checksum in `PIXELTRUTH_MODEL_SHA256`.
4. Start the app with `streamlit run app.py`.

---

## 🚀 Usage

```bash
# Run the Streamlit dashboard
streamlit run app.py
```

Then open your browser at `http://localhost:8501`, upload an image, and get instant results.

---

## 📁 Project Structure

```
PixelTruth/
├── app.py              # Streamlit dashboard
├── predict.py          # Inference logic
├── experiments/        # Model training & experiments
│   ├── train.py        # Model training (v1)
│   ├── train_v2.py     # Model training (v2)
│   └── train_v3.py     # Model training (v3)
├── scripts/            # Helper scripts
│   ├── download_model.py  # Model download helper
│   ├── create_dummy_model.py
│   └── check_merge_conflicts.py
├── requirements.txt    # Dependencies
├── Figure_1.png        # Training result plot
├── Figure_2.png        # Evaluation plot
└── README.md
```

---

## 🤝 Contributing

We welcome contributions of all kinds! Whether you're fixing a bug, improving the UI, or adding new features — you're welcome here.

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 🌸 GSSoC 2026

PixelTruth is participating in **GirlScript Summer of Code 2026**!

We have beginner-friendly issues ready. Look for issues labelled:
- `good-first-issue`
- `beginner-friendly`
- `documentation`
- `enhancement`

Feel free to explore open issues and start contributing!

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with ❤️ for media integrity by <a href="https://github.com/daksh-ugi">Piyush Sharma</a>
  <br/>
  If you found this useful, please ⭐ star the repo!
</p>

## ✨ README Improvement Notes

### 📌 Formatting Enhancements Needed
- Improve heading hierarchy for better readability
- Ensure consistent spacing between sections
- Use proper Markdown formatting for code blocks and lists
- Align all installation and usage steps properly

### 🚀 Suggested Structure Upgrade
- Introduction
- Features
- Tech Stack
- Installation
- Usage
- Project Structure
- Contribution Guidelines
- License

### 🛠️ Documentation Improvements
- Add badges (optional): build, license, contributors
- Add screenshots for better UI understanding
- Standardize code blocks for commands

### 🎯 Goal
Improve onboarding experience for new contributors and users by making README more structured, readable, and professional.

