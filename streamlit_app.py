import streamlit as st 
import pandas as pd 
import numpy as np 
from sklearn.neighbors import NearestNeighbors 
import pickle as pick 

st.set_page_config('Book Recommendations System')


st.title('Book Recommendations System') 

# Load the trained model 
try :
    with open('artifacts/model.pkl', 'rb') as f:
        model = pick.load(f) 

except Exception as e: 
    st.error(f'Failed to load the model: {str(e)}')

# Load the dataset 

try :
    popular_df=pd.read_csv(r'artifacts\ptdata.csv')
    pt = popular_df.pivot_table(values='Book-Rating',columns='User-ID',index='Book-Title').fillna(0) 
    pt_num = pt.to_numpy() 


except Exception as e: 
    st.error(f'Failed to load the dataset: {str(e)}')


def suggetion(book_name):
    sugg_books = []
    book_index = np.where(pt.index==book_name)[0][0]
    distances, indices = model.kneighbors([pt_num[book_index]])
    for i in indices[0][1:]:
        sugg_books.append(pt.index[i]) 
    return sugg_books 
    

# User input section 

user_input = st.text_input('Book Title') 

try :
    if user_input: 
        st.write('**Recommended Books:**')
        sugg_books = suggetion(user_input) 
        for book in sugg_books: 
            st.write(book) 
except Exception as e :
    st.error(f'Input book is not presnt')  
