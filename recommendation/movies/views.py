from django.shortcuts import render
from django.http import JsonResponse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
movie=pd.read_csv("C:/Users/Siva Ram/Desktop/intern/recommendation/movies/movie_recommendation.csv")
c=CountVectorizer(max_features=1000,stop_words='english')
v=c.fit_transform(movie['tags']).toarray()
cosine_similarity(v)
s=cosine_similarity(v)
titles_list =movie['title'].tolist()
def recommendations(m):

  try:
    m_i=movie[movie['title']==m].index[0]
    distances=s[m_i]
    l=sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]
    x=[]
    for i in l:
        x.append(movie['title'][i[0]])
    return x
  except Exception as e:    
    print(f"An error occurred: {e}")

  

# Create your views here.
def home(request):
    selected_title = request.GET.get('title', '')    
    context={'titles_list':titles_list,'selected_title': selected_title,'mv':recommendations(selected_title)}
    return render(request,'movies/home.html',context)
