# Multilingual Website Categorization with BERT

Welcome to our GitHub repository for Multilingual Website Categorization using BERT, a natural language processing (NLP) project aimed at predicting website categories across various languages. Below you'll find a detailed overview of the project files, technical insights, and how to get involved.

## Project Files:

 1. [**Bert_Model**](https://drive.google.com/drive/folders/14RhkZC76XNZEwEVSkYCYgzpAUZGJkj3K?usp=sharing)
  This folder contains the BERT model fine-tuned for project. Utilising the bert-multilingual-uncased model, it works effectively with 10+ languages to categorize websites accurately. Due to file size limits, I couldn't upload it directly to this repository. Instead, you can access the model through Google Drive.
   
2. **category_mapping.csv:**
   A CSV file with 50+ categories, including encoded values and their corresponding true labels.

3. **model_training.ipynb:**
   This notebook covers the entire process of training the NLP model. Sections include data loading, model training, and options for training from specific epochs.

4. **model_predictions.ipynb:**
   Here, you can find code for predicting website categories using the trained model. We've optimized it for efficiency, including multi-threading to handle large datasets.

5. **Website_Scraping.ipynb:**
   This notebook focuses on scraping data from websites, which is essential for training the model.

## Technical Insights:
Based on bert-multilingual-uncased model from the Transformers library, fine-tuned to suit our specific use case. The model achieves an accuracy of 92.7%, with challenges mainly arising in distinguishing between similar categories.

## Libraries Used:
- PyTorch: Used for building and training the neural network model.
- Transformers: Utilized for accessing pre-trained BERT models and fine-tuning them for our task.
- scikit-learn: Employed for data preprocessing, evaluation, and metrics calculation.
- BeautifulSoup: Utilized for web scraping, extracting text content from websites for training data.
- pandas: Used for data manipulation, particularly for handling CSV files and dataframes.
- numpy: Essential for numerical computations and array operations within the training and evaluation process.
- requests: Employed for making HTTP requests to websites during the web scraping process.
