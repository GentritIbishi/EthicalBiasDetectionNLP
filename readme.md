# Ethical Bias Detection in Kosovo-Serbia Relations Analysis and Conflict Mitigation on Twitter/X

### Academic Context  

 **University**:  `University of Prishtina`  
 **Faculty**: `Faculty of Electrical and Computer Engineering`  
 **Level**: `Master's Degree`   
 **Course**: `Natural Language Processing`  
 **Instructor**: `Prof.Dr.Sc. Mërgim Hoti`  
 **Authors**:  

   **MSc. (c) Gentrit Ibishi**  
   **MSc. (c) Guxim Selmani**  

  ---
## Project Overview

This repository contains an advanced Natural Language Processing (NLP) project focused on analyzing Kosovo-Serbia relations through social media discourse. The project implements sophisticated bias detection and conflict mitigation techniques to analyze Twitter/X data, providing insights into social media conversations surrounding this geopolitically sensitive topic.

## Research Objectives
1. Analyze social media discourse patterns
2. Detect and quantify various forms of bias
3. Implement mitigation strategies
4. Evaluate sentiment distribution
5. Provide data-driven insights into conflict-related discussions

## Technical Architecture  
### Development Environment    

* **Programming Language**: `Python`  
* **IDE**: `PyCharm Professional Edition`  
* **Version Control**: `Git`  
     * **Data Processing**: `Pandas, NumPy`   
     * **NLP Framework**: `Transformers, TextBlob`  
     * **ML**: `scikit-learn, imblearn`  
     * **Visualization**: `Matplotlib, Seaborn, WordCloud`   

### Dataset Specifications

* **Total Records**: `668 tweets`  
* **Primary Attributes**: `Tweet content`  
* **Source**: `Web scraping from Tweet API`  
* **Collection Period**: `November & December 2024`  
* **Focus**: `Kosovo-Serbia relations and associated conflicts`  

 
## Project structure
 <img width="540" alt="Bildschirmfoto 2024-12-21 um 23 11 15" src="https://github.com/user-attachments/assets/5123584b-fba1-48ef-be24-e056c4bd1789" />

---

## Implementation Components

### 1. Data Collection  
The project implements a robust Twitter data collection system with rate limiting and error handling:
![WebScrapingTweepyAPI.png](assets/WebScrapingTweepyAPI.png)  
   Key features:  
   * Multiple bearer token support  
   * Automatic rate limit handling with wait periods    
   * Batch processing to prevent memory overload  
   * Error handling and retry mechanisms  
   * "(Conflict Kosovo OR Conflict Serbia OR RKS OR SRB) lang:en -is:retweet"  
   
 
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
   
    
### 2. Data Preprocessing  

* URL and special character removal  
* Duplicate tweet removal (61 duplicates identified and removed)  
* Missing value handling  
* Text normalization and cleaning   
* Sentiment analysis using DistilBERT  

   ![img.png](assets/preprocessing.png)

     ```
    def _remove_urls_and_special_chars(self, text):
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#', '', text)
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    return text.strip()```

### 3. Bias Detection
   The system analyzes multiple types of bias:
   
#### Gender Bias Analysis  
   * Tracks usage of gendered pronouns  
   * Current distribution shows:  
      * Neutral pronouns: Highest usage (≈45%)  
      * Male pronouns: Moderate usage (≈10%)  
      * Female pronouns: Lowest usage (≈2%)  
      
        gender_pronouns = {
           'male': ['he', 'his', 'him'],
           'female': ['she', 'her', 'hers'],
           'neutral': ['they', 'their', 'them']
       }

#### Racial and Ethnic Term Analysis  
Monitors frequency of terms related to:  

* Kosovo/Albanian mentions: ~190 occurrences  
* Serbia/Serbian mentions: ~150 occurrences  
* Conflict-related terms: Significant presence  
      
       racial_terms = {
           'kosovo': ['kosovo', 'albanian', 'albanians'],
           'serbia': ['serbia', 'serbian', 'serbs'],
           'conflict-related': ['ethnic', 'minority', 'majority']
       }

   ![img.png](assets/biasdetection.png)

### 4. Bias Quantification
Measures bias through term frequency analysis:  

* Implements a conflict-peace term ratio analysis  
* Current bias balance: 0.741071  
* Tracks both conflict-related and peace-related terminology  
* Provides numerical metrics for bias assessment  

```def quantify_bias(self):
    conflict_keywords = ['war', 'violence', 'conflict', 'battle', 'fight', 
                        'struggle', 'invaded', 'killed']
    peace_keywords = ['peace', 'harmony', 'unity', 'calm', 'serenity', 
                     'non-violence']

    # Count occurrences
    self.tweets_df['conflict_count'] = self.tweets_df['Tweet'].apply(
        lambda x: sum(x.lower().count(word) for word in conflict_keywords)
    )
    self.tweets_df['peace_count'] = self.tweets_df['Tweet'].apply(
        lambda x: sum(x.lower().count(word) for word in peace_keywords)
    )

    # Calculate bias balance
    bias_balance = total_conflict / (total_peace + total_conflict)
```

   ![img.png](assets/quantifyingbias.png)

### 5. Bias Mitigation:  
  Implements content neutralization strategies:   

   -Text Content Neutralization
     ```def mitigate_bias(self):
      def neutralize_pronouns(text):
          for pronoun in ['he', 'him', 'his', 'she', 'her', 'hers']:
              text = text.replace(pronoun, 'they')
          return text
          
      def neutralize_racial_terms(text):
          replacements = {
              'kosovo': 'region',
              'serbia': 'region',
              'albanian': 'group',
              'serbian': 'group',
              'ethnic': 'community'
          }
          for term, replacement in replacements.items():
              text = re.sub(rf'\b{term}\b', replacement, text, flags=re.IGNORECASE)
          return text```
   

  ![img.png](assets/biasmitigation.png)

  

### 6.Sentiment Analysis: 
Distribution of sentiment across the dataset:

* Positive: ~320 tweets  
* Negative: ~280 tweets  
* Neutral: Minimal presence

### 7. Model Evaluation and Performance Analysis

Evaluating sentiment prediction using Logistic Regression and SMOTE for handling class imbalance.  
Uses Logistic Regression with SMOTE for balanced classification:  

   ![img.png](assets/evaluationmodel.png)


#### Implementation Details

```
def evaluation_model(self):
    # Data Preparation
    X = self.tweets_df['neutralized_text']
    y = self.tweets_df['sentiment'].apply(lambda x: 1 if x == 'POSITIVE' else 0)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Text Vectorization
    vectorizer = CountVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Handle Class Imbalance
    smote = SMOTE()
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_vec, y_train)

    # Model Training and Prediction
    model = LogisticRegression()
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test_vec)
```

#### Performance Metrics
Performance Metrics:  

* Accuracy: 80.33%  
* Precision: 0.82 (positive class)  
* Recall: 0.80 (positive class)  
* F1-Score: 0.80 (macro average)  
 


### 8. Visualization:  
Visualizing biases through bar plots, word clouds, and sentiment distributions.  

   ![img.png](assets/genderbias.png)

   ![img.png](assets/racialbias.png)

   ![img.png](assets/conflictbiasbalance.png)

   ![img.png](assets/sentimentdistribution.png)

   ![img.png](assets/wordsdistribution.png)


---

## Installation and Setup

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
4. Usage: Set up your Twitter API credentials in tweepyAPI.ipynb, update query parameters if needed & run it.
5. Open and run the main.ipynb file to perform the analysis

### Twitter API Requirements and Compliance
### Twitter API Registration Process
#### API Access Selection     
   <img width="250" alt="Bildschirmfoto 2024-12-01 um 14 01 31" src="https://github.com/user-attachments/assets/1773ddaf-eb0b-4186-9ce2-1771b5b1bb49" />    
 *Twitter's API pricing plans showing Basic and Pro access levels with their respective rate limits and features*    

#### Developer Agreement    
   <img width="250" alt="Bildschirmfoto 2024-12-01 um 14 03 52" src="https://github.com/user-attachments/assets/1f1b3e44-6951-4a75-8168-e33d397b52f8" />  
*Developer agreement form where we specified our academic research use case for ethical bias detection*  

**Usage Compliance**  
*  `Project registered as academic research`  
*  `Data collection focused on public tweets only`   
*  `Compliant with Twitter's developer terms`   
*  `No reselling or redistribution of Twitter data`   
*  `Multiple bearer tokens used for efficient rate limit management`   

