\# Emotion Classification: Naive Bayes vs Transformer



\## 1. Project Overview



This project focuses on classifying emotions expressed in text and comparing the performance of a traditional Bayesian machine learning approach with a modern Transformer-based model.



The project uses the DAIR-AI Emotion dataset and classifies text into six emotion categories:



\- Sadness

\- Joy

\- Love

\- Anger

\- Fear

\- Surprise



Two approaches are implemented:



1\. TF-IDF + Multinomial Naive Bayes

2\. DistilBERT Transformer



A Streamlit web application is also developed to allow users to enter text and obtain emotion predictions from both models.



\---



\## 2. Objectives



The main objectives of this project are:



\- To classify emotions from textual input.

\- To implement a traditional Bayesian classifier.

\- To implement a modern Transformer-based classifier.

\- To compare the performance of both approaches.

\- To evaluate the models using Accuracy, Macro F1-score and Weighted F1-score.

\- To develop an interactive Streamlit application for emotion prediction.



\---



\## 3. Dataset



The project uses the `dair-ai/emotion` dataset available through Hugging Face.



The dataset contains 20,000 text samples divided into:



\- Training: 16,000

\- Validation: 2,000

\- Testing: 2,000



The six emotion classes are:



| Label | Emotion |

|---|---|

| 0 | Sadness |

| 1 | Joy |

| 2 | Love |

| 3 | Anger |

| 4 | Fear |

| 5 | Surprise |



\---



\## 4. Methodology



\### 4.1 Data Preparation



The dataset is loaded and divided into training, validation and testing sets.



The text data is processed before being given to the machine learning model.



\### 4.2 Naive Bayes Model



The traditional machine learning approach uses:



\- TF-IDF Vectorization

\- Multinomial Naive Bayes



TF-IDF converts text into numerical feature vectors. These vectors are then used by the Multinomial Naive Bayes classifier to predict the emotion.



\### 4.3 Transformer Model



The modern approach uses a DistilBERT-based Transformer model:



`Sreekant13/distilbert-emotion`



The model is designed for the same six emotion classes used in this project.



The Transformer considers contextual relationships between words when making predictions.



\---



\## 5. Technologies Used



\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- Matplotlib

\- Seaborn

\- PyTorch

\- Hugging Face Transformers

\- Hugging Face Datasets

\- Streamlit



\---



\## 6. Experimental Results



The models were evaluated on the test dataset.



| Model | Accuracy | Macro F1 | Weighted F1 |

|---|---:|---:|---:|

| Naive Bayes | 69.40% | 44.00% | 63.00% |

| Transformer | 92.30% | 87.70% | 92.33% |



The Transformer achieved higher scores across the evaluated metrics on the test set used in this project.



\---



\## 7. Evaluation



The following evaluation methods were used:



\- Accuracy

\- Macro F1-score

\- Weighted F1-score

\- Confusion Matrix



The confusion matrices are available in the `results` folder.



\---



\## 8. Streamlit Application



The project includes an interactive Streamlit application.



The application allows users to:



\- Enter a text sentence.

\- Get an emotion prediction from Naive Bayes.

\- Get an emotion prediction from the Transformer.

\- View prediction confidence.

\- Compare the predictions of both models.

\- View probability distributions.

\- View the overall model performance.



\---



\## 9. How to Run the Project



\### Step 1: Clone or download the project



Open the project folder in VS Code or PowerShell.



\### Step 2: Create a virtual environment



```bash

python -m venv venv

