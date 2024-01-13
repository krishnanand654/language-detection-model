import streamlit as st
import pickle

with open('model.pkl','rb') as file:
    model = pickle.load(file)
   
with open('count_vectorizer.pkl','rb') as f:
    cv = pickle.load(f)

with open('label_encoder.pkl','rb') as fi:
    le = pickle.load(fi)

def main():
    st.title("Language detection")
    input_text = st.text_input("Enter the sentences:")
    if st.button('predict'):
        x = cv.transform([input_text]).toarray()
        prediction = model.predict(x)
        lang = le.inverse_transform([prediction])
        st.write(lang[0])
            
        





if __name__ == '__main__':
    main()