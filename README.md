# Sentiment_Analysis

<ul>
  <li>Project was aimed at rating a product based on Sentiment of large number of tweets corresponding to that product.</li>
  <li>Natural Language Processing was used to extract features in tweets.</li>
  <li>Sentiment was determined using various Machine Learning, Deep Learning and Dictionary based algorithms, which were compared on basis of different evaluation parameters.</li>
  <li>Skills– Python, Big Data, Natural Language Processing, Machine Learning and Deep Learning</li>
</ul>

<h2> Brief Project flow </h2>

<ul> 
	<li>Retrieving tweets content from twitter API</li>
	<li><b>Data Pre-processing:</b>
		<ul>
			<li>Removal Non Printable characters</li> 
			<li>Removal of URLs</li> 
			<li>Escaping HTML characters</li>
			<li>Split Attached words (DisplayIsAwesome -> Display Is Awesome)</li>
			<li>Emoji Replacer </L>
			<li>Replacing regular expressions (should've -> should have)</li>
			<li>Removal of unnecessary punctuation and tags
			<li>Tokenization</li>
			<li>Slang word Replacer (lol -> laugh out loud)</li>
			<li>Replacing of Abbreviations (A.S.A.P. -> as soon as possible)</li>
			<li>Replace repeating characters (happpppy -> happy)</li>
			<li>Checking Spellings</li>
			<li>Lemmatizing (eating -> eat)</li>
			<li>Replacing words with Contraction (do not uglify -> do beautify)</li>
			<li>Removal of Stop words</li>
			<li>Removal of Non Dictionary words</li>
			<li>Language Translation</li>
		</ul>
	<li><b>Text Feature Extraction:</b>
		<ul>
			<li>Representation of Bag of words using n-gram</li>
			<li>Normalizing and Weighting with diminishing importance of 
				<ul>
					<li>Tokens that are present in most of the samples and documents. </li>
					<li>Tokens with are very sparse</li>
				</ul>
			<li>Term Frequency-Inverse Document Frequency (TF-IDF)</li>
				<ul> 
					<li>Term Frequency (TF) = (Number of times term t appears in a document)/(Number of terms in the document)</li>
					<li>Inverse Document Frequency (IDF) = log(N/n), where, N is the number of documents and n is the number of documents a term t has appeared in.</li>
          <li>The IDF of a rare word is high, whereas the IDF of a frequent word is likely to be low. Thus having the effect of highlighting words that are distinct.</li>
					<li>We calculate TF-IDF value of a term as = TF * IDF</li>
				</ul>
			<li>Word2Vector to reconstruct linguistic context of words</li>
		</ul> 
<li><b>Algorithms for determining Sentiment of tweets:</b>
		<ul>
			<li>Supervised Learning</li>
				<ul>
					<li>Decision Tree</li>
					<li>Random Forest</li>
					<li>Gaussian Naive Bayes</li>
					<li>Multinomial Naive Bayes</li>
					<li>Support Vector Classification</li>
					<li>Logistic Regression</li>
					<li>Neural Network</li>
					<li>Multi Layer perceptron</li>
				</ul>
			<li>Lexicon Based</li>
				<ul>
					<li><b>Dictionary based:</b> Dictionary based sentiment analysis is based on comparison between the text or corpus with pre-established dictionaries of positive, negative and neutral words.</li>
					<li><b>Dictionary based with Score:</b> Sentiment score of a tweet is given by the sum of positive and negative ratings of words in it.</li>
				</ul>
		</ul>
<li><b>Evaluation Parameters:</b>
		<ul> 
			<li>Accuracy</li>
			<li>Precision</li>
			<li>F1Score</li>
			<li>Recall</li>
			<li>Cohen Kappa</li>
			<li>Hamming Loss</li>
			<li>Jaccard Similarity</li>
			<li>Execution Time</li>
		</ul>
	<li><b>Application: </b>Using Sentiment Classification model to give ratings to various movies and products. 
</ul>
