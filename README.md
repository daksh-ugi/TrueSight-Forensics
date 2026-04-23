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

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/daksh-ugi/PixelTruth.git
cd PixelTruth

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

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
├── train.py            # Model training (v1)
├── train_v2.py         # Model training (v2)
├── train_v3.py         # Model training (v3)
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
```
