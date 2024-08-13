Practical 5A
from indicnlp.tokenize import indic_tokenize

# Sample Hindi text
hindi_text = "यह एक उदाहरण वाक्य है।"

# Tokenize the Hindi text using indic-nlp-library
tokens = indic_tokenize.trivial_tokenize(hindi_text)

# Display the tokens
print('Word Tokens:', tokens)
















Practical 5B
# Dictionary containing synonyms
synonyms = {
    "खुश": ["प्रसन्न", "आनंदित", "खुशी"],
    "बहुत": ["अधिक", "बहुत ज्यादा", "काफी"]
}

# Function to generate similar sentences by replacing some words with synonyms
def generate_similar_sentences(input_sentence, num_sentences=5):
    similar_sentences = []
    # Replace some words with synonyms 
    for word, word_synonyms in synonyms.items():
        for synonym in word_synonyms:
            modified_sentence = input_sentence.replace(word, synonym)
            similar_sentences.append(modified_sentence)
    return similar_sentences[:num_sentences]

# Input sentence
input_sentence = "मैं आज बहुत खुश हूँ।"
# Generate similar sentences
similar_sentences = generate_similar_sentences(input_sentence)
# Output the results
print("Original sentence:", input_sentence)
print("Similar sentences:")
for sentence in similar_sentences:
    print("-", sentence)
 
Practical 5C
import nltk
import langid
nltk.download('punkt')
def indetify_language(text):
    lang,_ = langid.classify(text)
    return lang
language = indetify_language("नमस्ते, आप कै से हैं?")
language2 = indetify_language("এটি একটি উদাহরণ বাক্য।")
print("Identified language:", language)
print("Identified language:", language2)
 
Practical 6A
import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
from nltk import tokenize
from nltk import tag
from nltk import chunk
para = "Natural language processing (NLP) is a machine learning technology that gives computers the ability tointerpret, manipulate, and comprehend human language."
sents = tokenize.sent_tokenize(para)
print("\nsentence tokenization\n===================\n",sents)
# word tokenization
print("\nword tokenization\n===================\n")
for index in range(len(sents)):
 words = tokenize.word_tokenize(sents[index])
 print(words)
# POS Tagging
tagged_words = []
for index in range(len(sents)):
 tagged_words.append(tag.pos_tag(words))
print("\nPOS Tagging\n===========\n",tagged_words)
# chunking
tree = []
for index in range(len(sents)):
 tree.append(chunk.ne_chunk(tagged_words[index]))
print("\nchunking\n========\n")
print("Tree: ",tree)
Practical 6B
#pip install -U spacy
#python -m spacy download en_core_web_sm
import spacy
# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_sm")
# Process whole documents
text = ("Natural language processing (NLP) is an interdisciplinary subfield of computer science and information retrieval. It is primarily concerned with giving computers the ability to support and manipulate human language. It involves processing natural language datasets, such as text corpora or speech corpora, using either rule-based or probabilistic machine learning approaches.")
print("Original text: ", text, "\n")
doc = nlp(text)
# Analyse syntax
print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"]) 
Practical 6C
import nltk 
nltk.download('treebank') 
from nltk.corpus import treebank_chunk 
treebank_chunk.tagged_sents()[0] 
treebank_chunk.chunked_sents()[0] 
treebank_chunk.chunked_sents()[0].draw()
 
Practical 7A
import nltk
from nltk import tokenize
grammar1 = nltk.CFG.fromstring("""
S -> VP
VP -> VP NP
NP -> Det NP
Det -> 'that'
NP -> 'flight'
VP -> 'Book'
""")
sentence = "Book that flight"
all_tokens = tokenize.word_tokenize(sentence)
print(all_tokens)
parser = nltk.ChartParser(grammar1)
for tree in parser.parse(all_tokens):
 print(tree)
 tree.draw()
 
Practical 7B
def FA(s):
    # If the length is less than 3, then it can't be accepted. Therefore, end the process.
    if len(s) < 3:
        return "Rejected"
    
    if s[0] == '1':
        if s[1] == '0':
            if s[2] == '1':
                for i in range(3, len(s)):
                    if s[i] != '1':
                        return "Rejected"
                return "Accepted"  # If all 4 nested if statements are true
            return "Rejected"  # else of 3rd if
        return "Rejected"  # else of 2nd if
    return "Rejected"  # else of 1st if

# Input strings to test the FA function
inputs = ['1', '10101', '101', '10111', '01010', '100', '', '10111101', '1011111']

# Test the FA function with the provided inputs
for i in inputs:
    print(FA(i))




Practical 7C
def FA(s):
    size = 0
    # Scan the complete string and make sure that it contains only 'a' & 'b'
    for i in s:
        if i == 'a' or i == 'b':
            size += 1
        else:
            return "Rejected"
    # After checking that it contains only 'a' & 'b'
    # Check its length; it should be at least 3
    if size >= 3:
        # Check the last 3 elements
        if s[size - 3] == 'b':
            if s[size - 2] == 'b':
                if s[size - 1] == 'a':
                    return "Accepted"  # if all 4 if conditions are true
                return "Rejected"  # else of 4th if
            return "Rejected"  # else of 3rd if
        return "Rejected"  # else of 2nd if
    return "Rejected"  # else of 1st if
# Input strings to test the FA function
inputs = ['bba', 'ababbba', 'abba', 'abb', 'baba', 'bbb', '']
# Test the FA function with the provided inputs
for i in inputs:
    print(FA(i)) 
Practical 7D
#!pip install nltk
#nltk.download('punkt')
import nltk
from nltk import tokenize
grammar1 = nltk.CFG.fromstring("""
 S -> NP VP
 PP -> P NP
 NP -> Det N | Det N PP | 'I'
 VP -> V NP | VP PP
 Det -> 'a' | 'my'
 N -> 'bird' | 'balcony'
 V -> 'saw'
 P -> 'in'
 """)
sentence = "I saw a bird in my balcony"
for index in range(len(sentence)):
 all_tokens = tokenize.word_tokenize(sentence)
print(all_tokens)
# all_tokens = ['I', 'saw', 'a', 'bird', 'in', 'my', 'balcony']
parser = nltk.ChartParser(grammar1)
for tree in parser.parse(all_tokens):
 print(tree)
 tree.pretty_print()


