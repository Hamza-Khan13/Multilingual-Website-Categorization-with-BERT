# Multilingual Website Categorization with BERT

Welcome to our GitHub repository for Multilingual Website Categorization using BERT, a natural language processing (NLP) project aimed at predicting website categories across various languages. Below you'll find a detailed overview of the project files, technical insights, and how to get involved.

## Project Files:

1. [**Bert_Model**](https://drive.google.com/drive/folders/14RhkZC76XNZEwEVSkYCYgzpAUZGJkj3K?usp=sharing)
  This folder contains the BERT model I've fine-tuned for my project. Using bert-multilingual-uncased, it works with 10+ languages to categorize websites accurately. It's the core of my project, processing website data and predictions effectively.
  Since I could not upload it here due to file size limits, I have attached a Google Link.
   
3. **category_mapping.csv:**
   A CSV file with 50+ categories, including encoded values and their corresponding true labels.

4. **model_training.ipynb:**
   This notebook covers the entire process of training the NLP model. Sections include data loading, model training, and options for training from specific epochs.

5. **model_predictions.ipynb:**
   Here, you can find code for predicting website categories using the trained model. We've optimized it for efficiency, including multi-threading to handle large datasets.

6. **Website_Scraping.ipynb:**
   This notebook focuses on scraping data from websites, which is essential for training the model.

## Technical Insights:
Based on bert-multilingual-uncased model from the Transformers library, fine-tuned to suit our specific use case. The model achieves an accuracy of 92.7%, with challenges mainly arising in distinguishing between similar categories.
