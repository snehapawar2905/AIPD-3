# 🤖 AI-Based Product Development – Practical 3

## 📌 Project Title

**Sentiment Analysis of Customer Reviews using Deep Learning**

## 📖 Project Description

This project focuses on developing a **Sentiment Analysis AI model** to analyze customer reviews of products or services. The goal is to automatically classify customer opinions as **positive or negative**, helping businesses understand customer satisfaction and improve their offerings.

The model is implemented using **deep learning techniques** and trained on a dataset of customer reviews. This practical demonstrates how Natural Language Processing (NLP) can be applied to real-world business problems.

## 🎯 Objective

* Develop a sentiment analysis model for customer reviews
* Apply NLP techniques for text preprocessing
* Train the model using a deep learning framework
* Evaluate model performance using classification metrics

## 🧠 Use Case

**Customer Review Sentiment Analysis**

* Input: Customer review text
* Output: Sentiment classification (Positive / Negative)

## 🛠️ Tech Stack

* **Language:** Python
* **Framework:** TensorFlow / PyTorch
* **Libraries:**

  * NumPy
  * Pandas
  * Scikit-learn
  * NLTK
  * Torch / TensorFlow

## 📊 Dataset Description

* Dataset contains customer reviews and sentiment labels
* Labels:

  * 1 → Positive Review
  * 0 → Negative Review
* Text preprocessing steps include:

  * Lowercasing
  * Tokenization
  * Stopword removal
  * Lemmatization
  * Vectorization (TF-IDF / Embeddings)
  * 
## 🧪 Model Architecture

The sentiment analysis model includes:

* Text vectorization layer
* Hidden layers with ReLU activation
* Output layer with Sigmoid activation
* Loss Function: **Binary Cross-Entropy**
* Optimizer: **Adam**

## 🚀 Model Training

* Dataset split into training and testing sets
* Model trained over multiple epochs
* Loss minimized using gradient descent
* Validation performed to monitor performance

## 📈 Model Evaluation

The model performance is evaluated using:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* **Confusion Matrix**

Sample Result:

```
Accuracy: ~88% – 92%
```

## 🔍 Results and Analysis

* Model effectively classifies customer sentiments
* High accuracy indicates good generalization
* NLP preprocessing significantly improves performance

## 🧩 Future Enhancements

* Multi-class sentiment classification (Positive / Neutral / Negative)
* Use of LSTM / GRU / Transformer models
* Real-time sentiment analysis dashboard
* Integration with e-commerce platforms
* Deployment using Flask or FastAPI


## ▶️ How to Run the Project

1. Install required libraries:

```bash
pip install pandas numpy scikit-learn nltk torch tensorflow
```

2. Run the notebook:

```bash
jupyter notebook "SP AIPD Prac3.ipynb"
```

## 📌 Conclusion

This project successfully demonstrates the development of an AI-based sentiment analysis system for customer reviews. It highlights the importance of NLP and deep learning in understanding customer feedback and supporting data-driven business decisions.

## 👩‍💻 Author

**Pawar Sneha Sachin**
TY-AI&DS
