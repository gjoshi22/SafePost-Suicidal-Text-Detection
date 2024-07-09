#Mental Health Counselor Chatbot

A Chatbot Trained on Mental Health Q&A data between doctors and patients was developed by fine-tuning Meta’s LLAMA2 large-language model using the QLORA technique to guide individuals towards appropriate therapeutic directions. The trained chatbot has been uploaded to hugging face library for anyone to try and use.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Preprocessing](#preprocessing)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Model Building](#model-building)
    - [Logistic Regression](#logistic-regression)
    - [CNN](#cnn)
    - [LSTM](#lstm)
    - [BERT](#bert)
    - [ELECTRA](#electra)
- [Results](#results)
- [Conclusion](#conclusion)
- [Usage](#usage)
- [References](#references)

## Introduction

The mental health crisis among adolescents has reached alarming proportions, with 1 in 5 reporting symptoms of anxiety and depression. This project aims to address this issue by developing a predictive model to detect posts containing suicidal ideation on social media platforms such as Reddit.

## Dataset

The dataset used for the project was collected from Kaggle, containing posts from two subreddits: r/SuicideWatch and r/depression. The dataset includes 232,074 rows and 2 columns: "text" and "class". The "class" column indicates whether the post is labeled as "suicide" or "non-suicide". The dataset can be accessed [here](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch).

## Methodology

### Preprocessing

To prepare the dataset for model training, several preprocessing steps were applied:

1. **Spell Check:** Used the SymSpell library to correct spelling mistakes. This step ensures the text is accurate and reduces noise in the data.
2. **Remove Stop Words:** Essential words such as "not," "never," and "but" were retained to ensure accurate sentiment analysis, while other common stop words were removed to reduce dimensionality.
3. **Remove Extra White Spaces:** Trimmed leading/trailing whitespaces and replaced multiple spaces within the text to maintain a consistent format.
4. **Remove Diacritics:** Converted accented characters to their unaccented equivalents, improving the uniformity of the text data.
5. **Remove URL:** Used regular expressions to eliminate URLs from the text, cleaning unnecessary links and potential noise.
6. **Remove Symbols and Digits:** Excluded non-alphabetic characters to focus solely on textual information.
7. **Fix Lengthening of Words:** Normalized repeated characters using regular expressions to correct lengthened words (e.g., "finallllllly" to "finally").
8. **Lemmatization:** Converted tokens to their base forms using the Spacy NLP library, standardizing the text for better analysis and comparison.

### Exploratory Data Analysis

To understand the dataset better and uncover hidden patterns, several exploratory data analysis (EDA) techniques were employed:

1. **Word Cloud:** Generated word clouds for suicidal and non-suicidal texts to visualize the most frequently occurring words and gain insights into the common themes.
2. **Average Text Length Distribution:** Analyzed the distribution of text lengths using histograms, revealing that suicidal texts tend to be longer and contain more detailed expressions.
3. **Frequent Bigrams:** Identified common bigrams in the dataset, which provided context for understanding recurring word pairs and their associations with suicidal ideation.

### Model Building

Multiple models were built and evaluated to classify texts containing suicidal ideation. Each model was designed with specific architectures and hyperparameters:

#### Logistic Regression

A baseline model using Logistic Regression was implemented with custom Word2Vec embeddings. This model served as a simple yet effective approach to classify the text data.

#### CNN

A Convolutional Neural Network (CNN) was built to capture local textual patterns using custom Word2Vec embeddings. The CNN applied convolutional filters to extract relevant features, followed by a max-pooling layer to reduce dimensionality.

#### LSTM

Long Short-Term Memory (LSTM) networks were employed to capture long-term dependencies in the text. The LSTM architecture included multiple layers with dropout regularization to prevent overfitting and enhance model generalization.

#### BERT

The Bidirectional Encoder Representations from Transformers (BERT) model was fine-tuned on the dataset. BERT's bidirectional nature allowed it to understand the context of words from both directions, resulting in improved performance in text classification tasks.

#### ELECTRA

ELECTRA model utilized a unique pre-training process called Replaced Token Detection (RTD). This approach involved a generator model to replace masked tokens and a discriminator model to identify real and fake tokens, leading to a more accurate representation of the text data.

## Results

| Model                | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression  | 0.9054   | 0.9055    | 0.9054 | 0.9055   |
| CNN                  | 0.9224   | 0.9059    | 0.8902 | 0.8941   |
| LSTM                 | 0.9234   | 0.9093    | 0.8876 | 0.8941   |
| BERT                 | 0.9752   | 0.9678    | 0.9677 | 0.9678   |
| ELECTRA              | 0.9761   | 0.9690    | 0.9685 | 0.9687   |

## Conclusion

The study demonstrates the effectiveness of various NLP models in detecting suicidal ideation in social media posts. The ELECTRA model achieved the highest accuracy and performance metrics, underscoring its proficiency in handling text classification tasks. The results highlight the potential of transformer-based models in understanding complex linguistic patterns and identifying at-risk individuals based on their online expressions. This project provides a foundation for further research and development of AI-driven mental health support systems, emphasizing the importance of early detection and intervention in addressing mental health crises.

## Usage

The trained chatbot model has been uploaded to Hugging Face for public use. You can access it [here](https://huggingface.co/gunjanjoshi/llama2-7b-sharded-bf16-finetuned-mental-health-conversational).

To use the fine-tuned model locally, you can refer to the notebook provided [here](https://github.com/gjoshi22/SafePost-Suicidal-Text-Detection/blob/main/finetuned_qlora_llama2_7b_sharded_final.ipynb).

## References

1. National Institute of Mental Health
2. Kaggle Dataset: [Suicide Watch](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch)
3. SymSpell library
4. Google Word2Vec
5. Hugging Face Transformers
