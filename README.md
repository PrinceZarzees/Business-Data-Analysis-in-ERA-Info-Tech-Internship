# Business-Data-Analysis-in-ERA-Info-Tech-Internship
We want to classify our customers transaction Narration data to understand the pattern and purposes of transaction. Our aim is to identify the behavioral pattern of customer from their transaction statement whether it is a deposit type transaction, a purchase type transaction, or a credit type transaction etc. So, we want a to build a trained NLP model which can create clusters of transaction narration where each segment contains similar type of transaction narratives.
# Summary
Performed some basic preprocessing techniques of NLP like tokenization, stop word removal, punctuation removal, digit removal, stemming, and lemmatization.  
NLP Pipeline:  
<ol>
  <li>Text Preprocessing</li>    
  <li>Text Cleaning </li>  
  <li>Lowercasing</li>  
  <li>Special character removal</li>  
  <li>Punctuation removal etc.</li>  
</ol>  
<ol>
<li>Tokenization on cleaned text</li>  
  <li>Normalization on Tokens  
  <ul><li> Stemming </li>  
    <li> Lemmatization </li>
   </ul>
   </li>
  <li>Stop words removal</li>  
<li>Keyword Extraction  
  <ul>
    <li>TF-IDF</li></ul>  
  </li>
<li>Featured Engineering/ Feature Extraction (Converting text data to numerical TF-IDF Vectorizer)</li>  
  <li>Clustering using K-means and Latent Dirichlet Allocation (LDA)</li>
</ol>
  
# Running the model  
First clone the repository to your desired folder. Open a terminal and write:  
```
python3 test_script.py <text> <method> 
```
Here text is the string I want to get the cluster and method is kmeans or lda  
For example
```
  python3 test_script.py "cash withdraw" "kmeans"
```
Output:  
```
Preprocessed Text: cash withdraw
Cluster: 6
```
  
  
  
  
  
  
  
