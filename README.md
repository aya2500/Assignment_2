# 🚀 Assignment 2: Debugging Vision Transformers (ViT) with PyCharm & WSL

## 👋 Introduction
This project demonstrates step-by-step debugging and inspection of the Vision Transformer (ViT-B/16) using **PyCharm** + **WSL (Ubuntu 24.04)**. All tensor shapes and values are visualized as the model progresses from raw image input to classification logits.

---

## 🖥️ Environment
- 💻 **IDE:** PyCharm Professional
- 🐧 **Platform:** WSL (Ubuntu 24.04)
- 🎨 **Model:** ViT-B/16  
    - 🧩 Patch size: 16  
    - 🧠 Embedding dimension: 768  
    - 🔮 Attention heads: 12  
    - 🏛️ Encoder blocks: 12  
    - 🏷️ Classification head for 1000 ImageNet classes

---

## 🖼️ Input Image & Preprocessing
- 📷 **Selected Image:** `sample.jpg` (personal photo)
- ⚡ **Preprocessing Steps:**
    1. 🌈 Convert to RGB
    2. ✂️ Resize to 224 × 224
    3. ⚖️ Normalize with ImageNet means `[0.485, 0.456, 0.406]` & std `[0.229, 0.224, 0.225]`
    4. 🧬 Convert to tensor `(1, 3, 224, 224)`

---

## 🦾 Model Trace: Snapshots

| 🪜 **Step**                | 🧩 **Shape**       | 📝 **Description**               |
|---------------------------|-------------------|----------------------------------|
| Raw Input                 | `[1, 3, 224, 224]`  | Raw image after preprocessing    |
| 🚦 Patchification         | `[1, 768, 14, 14]`  | Split into 16×16 patches         |
| 🧹 Flattened Patches      | `[1, 196, 768]`     | 196 patches as 768-dim vectors   |
| 🪄 Patch Embeddings       | `[1, 196, 768]`     | Linear projection                |
| 🏷️ Class Token            | `[1, 1, 768]`       | Learnable classification token   |
| ➕ Embeddings + Class      | `[1, 197, 768]`     | Class token prepended            |
| 🔢 + Positional Encoding  | `[1, 197, 768]`     | Adds spatial info                |
| 🛕 Encoder Input           | `[1, 197, 768]`     | Start of first block             |
| 🔍 Q, K, V                | `[1, 197, 768]`     | Multi-head attention projections |
| 🧮 Attention Scores        | `[1, 12, 197, 197]` | Before/after softmax             |
| 🔗 Residual + Norm        | `[1, 197, 768]`     | Output of attention block        |
| 🧑‍💻 MLP Hidden            | `[1, 197, 3072]`    | Feed-forward network output      |
| 🏁 Encoder Output          | `[1, 197, 768]`     | After 1/2/12 blocks              |
| 🏷️ Class Token            | `[1, 768]`          | Extracted for classification     |
| 🚨 Logits                 | `[1, 1000]`         | Projected to ImageNet classes    |
| 🧮 Softmax                | `[1, 1000]`         | Probability for each class       |

---

## ❓ Key Questions

- **🧩 Why split images into patches?**  
  > Transformers require sequences. Patches convert 2D images to 1D token sequences.

- **🏷️ Why add a class token?**  
  > Aggregates patch info. Classification uses its representation.

- **🔢 Why positional encodings?**  
  > Provide spatial info since transformers are permutation-invariant.

- **🔍 Why Q, K, V with same dimensions?**  
  > Allows matrix operations for attention, compatible with number of tokens.

- **🔗 Why do residuals preserve shape?**  
  > Stabilizes training for deep stacks.

- **🏷️ Why only the class token for classification?**  
  > Serves as summary of all patches for final prediction.

---

## 🌈 Reflection
Debugging step-by-step with PyCharm truly unlocked my understanding of the Vision Transformer architecture — seeing how data flows, how attention works, and appreciating the design choices (patches, class token, positional encoding).

---

## 🏃‍♀️ How to Run

1. Open project in **PyCharm**.
2. Choose WSL as the Python interpreter.
3. Put `sample.jpg` in the project folder.
4. Run or debug **`main3.py`**.

---

## 🗂️ File Structure

Assignment_2/
├── main3.py # Main ViT debugging script
├── sample.jpg # Input image for testing
└── README.md # Project documentation


---

## 📚 License

This project is part of a university assignment. For educational use only.
