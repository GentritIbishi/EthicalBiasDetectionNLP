# Ethical Bias Detection in Kosovo-Serbia Relations Analysis and Conflict Mitigation on Twitter/X

### Description

This repository contains a project focused on analyzing Kosovo-Serbia relations and mitigating conflicts on Twitter/X. The project involves collecting data, processing and analyzing tweets to detect biases, quantify conflicts, and implement techniques for conflict mitigation.

### University, Faculty, Level of Study, Course, and Instructor

- **University**:  University of Prishtina
- **Faculty**: Faculty of Electrical and Computer Engineering
- **Level of Study**: Master  
- **Course**: Natural Language Processing
- **Instructor**: Prof.Dr.Sc. Mërgim Hoti 

### Authors

- **MSc. (c) Gentrit Ibishi**  
- **MSc. (c) Guxim Selmani**

  ---

### Development Environment  

- **Programming Language**: `Python` 
- **IDE**: `PyCharm Professional Edition` 
- **Version Control**: `Git`
 

### Dataset Details

- **Total Rows**: `668`
- **Attributes**: `Tweet`  
- **Source**: `Web scraping from Tweet API`
- **Query Parameters**: `Focused on Kosovo-Serbia relations and conflicts`
- **Data Collection Period**: `November & December 2024`

 
### Project structure
 <img width="540" alt="Bildschirmfoto 2024-12-21 um 23 11 15" src="https://github.com/user-attachments/assets/5123584b-fba1-48ef-be24-e056c4bd1789" />

---

## Required Libraries and Their Functions  
### Core Data Processing Libraries  
- Pandas  
  Purpose: Data manipulation and analysis  
  ```
  # DataFrame operations
   df.dropna(subset=['Tweet'])  # Remove missing values
   df.drop_duplicates()         # Remove duplicate tweets
   df['new_column'] = df['Tweet'].apply(function)  # Apply transformations
  ```
- NumPy  
  ```
  Purpose: Numerical computations
  ```
 - LogisticRegression  

### Natural Language Processing Libraries  
- Transformers  
  Purpose: State-of-the-art NLP models  
  Implementation:
  ```
  from transformers import pipeline
   sentiment_pipeline = pipeline('sentiment-analysis', 
                            model='distilbert/distilbert-base-uncased-finetuned-sst-2-english')
  ```
- TextBlob
  Purpose: Text processing and sentiment analysis  
  Features: Sentiment scoring, subjectivity analysis  

### Visualization Libraries
- Matplotlib & Seaborn  
  ```
  def visualize_analysis(self):
    plt.figure(figsize=(8, 6))
    sns.barplot(data=gender_df, x='Gender', y='Bias', palette='pastel')
    plt.title('Gender Bias Analysis')
  ```  
- wordcloud  
  Purpose: Generate word frequency visualizations
  ```
  wordcloud = WordCloud(stopwords='english', 
                     background_color='white').generate(' '.join(texts))
  ```
### Machine Learning Libraries
- scikit-learn  
Components used:  
CountVectorizer: Text vectorization  
train_test_split: Data splitting  
LogisticRegression: Classification  
metrics: Performance evaluation  

- imblearn  
  Purpose: Handle imbalanced datasets  
  Implementation:  
  ```
  smote = SMOTE()
   X_train_resampled, y_train_resampled = smote.fit_resample(X_train_vec, y_train)
  ```
---

## Project Overview  

This project implements a NLP-based system for analyzing and mitigating biases in social media discourse surrounding Kosovo-Serbia relations. The system processes Twitter data to detect various forms of bias, quantify conflicts, and implement conflict mitigation strategies through advanced natural language processing techniques. It includes the following steps:

1. **Data Collection**: The project uses a robust Twitter data collection system implemented in tweepy_api.ipynb including key features:
   
   *Multiple bearer token support for extended rate limits*  
   *Automatic rate limit handling with wait periods*  
   *Batch processing to prevent memory overload*  
   *Error handling and retry mechanisms*  
   
   Usage Compliance

  `Project registered as academic research`  
  `Data collection focused on public tweets only`   
  `Compliant with Twitter's developer terms`   
  `No reselling or redistribution of Twitter data`   
  `Multiple bearer tokens used for efficient rate limit management`   
     <img width="250" alt="Bildschirmfoto 2024-12-01 um 14 03 52" src="https://github.com/user-attachments/assets/1f1b3e44-6951-4a75-8168-e33d397b52f8" />
     <img width="250" alt="Bildschirmfoto 2024-12-01 um 14 01 31" src="https://github.com/user-attachments/assets/1773ddaf-eb0b-4186-9ce2-1771b5b1bb49" />  

   
![WebScrapingTweepyAPI.png](assets/WebScrapingTweepyAPI.png)

   ```
   # Twitter API configuration
   BEARER_TOKENS = [
    "TOKEN1",
    "TOKEN2",
    "TOKEN3"
   ]

   # Configurable query parameters
   queries = [
    '(Conflict Kosovo OR Conflict Serbia OR RKS OR SRB) lang:en -is:retweet'
   ]
   ```
    
3. **Data Preprocessing**: Comprehensive text preprocessing implemented in the EthicalBiasDetection class. Cleaning and preprocessing data by removing URLs, special characters, and performing sentiment analysis.

       #  def _remove_urls_and_special_chars(self, text):
       # Remove URLs
       text = re.sub(r'https?://\S+|www\.\S+', '', text)
       # Remove mentions and hashtags
       text = re.sub(r'@\w+|#', '', text)
       # Remove non-alphabetic characters
       text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
       return text.strip()
  

   ![img.png](assets/preprocessing.png)

4. **Bias Detection**: Detecting gender, racial, political, and other biases using advanced natural language processing techniques. 

       # def detect_bias(self):
       # Gender bias detection
       gender_pronouns = {
           'male': ['he', 'his', 'him'],
           'female': ['she', 'her', 'hers'],
           'neutral': ['they', 'their', 'them']
       }
       
       # Racial and ethnic term analysis
       racial_terms = {
           'kosovo': ['kosovo', 'albanian', 'albanians'],
           'serbia': ['serbia', 'serbian', 'serbs'],
           'conflict-related': ['ethnic', 'minority', 'majority']
       }

   ![img.png](assets/biasdetection.png)

5. **Quantifying Bias**: Quantifying bias by counting conflict-related and peace-related terms in tweets.

   ![img.png](assets/quantifyingbias.png)

6. **Mitigation**: Neutralizing gender and racial biases through text manipulation.

      ![img.png](assets/biasmitigation.png)

7. **Visualization**: Visualizing biases through bar plots, word clouds, and sentiment distributions.

   ![img.png](assets/genderbias.png)

   ![img.png](assets/racialbias.png)

   ![img.png](assets/conflictbiasbalance.png)

   ![img.png](assets/sentimentdistribution.png)

   ![img.png](assets/wordsdistribution.png)

8. **Model Evaluation**: Evaluating sentiment prediction using Logistic Regression and SMOTE for handling class imbalance.

   ![img.png](assets/evaluationmodel.png)

---

## Installation and Usage

### Prerequisites
  
  `Python 3.x`  
  `PyCharm IDE`    
  `Twitter API credentials`  

### Setup Steps

1. Clone the repository:  
   ```
   [git clone https://github.com/yourusername/kosovo-serbia-analysis.git]
2. Install required packages  
   `pip install pandas numpy tweepy scikit-learn transformers textblob wordcloud seaborn matplotlib imblearn`
4. Usage  
   Set up your Twitter API credentials in tweepyAPI.ipynb, update query parameters if needed & run it.
5. Open and run the main.ipynb file to perform the analysis
