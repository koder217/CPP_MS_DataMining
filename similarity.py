# -------------------------------------------------------------------------
# AUTHOR: Siva Charan Mallena
# FILENAME: similarity.py
# SPECIFICATION: Cosine Similarity exercise as part of assignment 1
# FOR: CS 5990 (Advanced Data Mining) - Assignment #1
# TIME SPENT: ~1 hr
# -----------------------------------------------------------*/

# Importing some Python libraries
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# Defining the documents
doc1 = "Soccer is my favorite sport"
doc2 = "I like sports and my favorite one is soccer"
doc3 = "I support soccer at the olympic games"
doc4 = "I do like soccer, my favorite sport in the olympic games"



# Use the following words as terms to create your document matrix
terms = ['soccer', 'my', 'favorite', 'sport', 'I', 'like', 'one', 'support', 'olympic', 'games']
matrix = []
def get_key_matrix(terms, doc):
    words = doc.split()
    arr = []
    for word in words:
        for term in terms:
            if term.lower() == word.lower():
                arr.append(word.lower())
    return arr

doc1Arr = get_key_matrix(terms, doc1)
doc2Arr = get_key_matrix(terms, doc2)
doc3Arr = get_key_matrix(terms, doc3)
doc4Arr = get_key_matrix(terms, doc4)

doc1_vals = Counter(doc1Arr)
doc2_vals = Counter(doc2Arr)
doc3_vals = Counter(doc3Arr)
doc4_vals = Counter(doc4Arr)

# print(doc1_vals)
# print(doc2_vals)
# print(doc3_vals)
# print(doc4_vals)

words  = list(doc1_vals.keys() | doc2_vals.keys() | doc3_vals.keys() | doc4_vals.keys())
d1_vect = [doc1_vals.get(word, 0) for word in words]       
d2_vect = [doc2_vals.get(word, 0) for word in words]
d3_vect = [doc3_vals.get(word, 0) for word in words]       
d4_vect = [doc4_vals.get(word, 0) for word in words]

matrix.append(d1_vect)
matrix.append(d2_vect)
matrix.append(d3_vect)
matrix.append(d4_vect)

#print(matrix)
#print(cosine_similarity([d1_vect], [d2_vect]))
#print(cosine_similarity([d1_vect,d2_vect, d3_vect, d4_vect]))

# Compare the pairwise cosine similarities and store the highest one
# Use cosine_similarity([X], [Y]) to calculate the similarities between 2 vectors only
# Use cosine_similarity([X, Y, Z]) to calculate the pairwise similarities between multiple vectors
max_score = 0
best_match = []
for id1, x in enumerate(matrix):
    for id2, y in enumerate(matrix):
        similarity = cosine_similarity([x], [y])[0][0]
        #print(id1, id2, similarity)
        if id1 != id2 and similarity > max_score:
            best_match = []
            best_match.append(id1+1)
            best_match.append(id2+1)
            best_match.append(similarity)
            max_score = similarity

#   PS: THE LOOP ABOVE HAS REDUNDANT ITERATIONS. 
#   COMPARING DOC1 , DOC2 IS SAME AS DOC2 AND DOC1.
#   OMITTING OPTIMIZATIONS TO KEEP IT SIMPLE AND FOR BREVITY.


# Print the highest cosine similarity following the template below
# The most similar documents are: doc1 and doc2 with cosine similarity = x
print('most similar documents are: doc'+str(best_match[0])+' and doc'+str(best_match[1])+' with cosine similarity = '+str(best_match[2]))
