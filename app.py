import streamlit as st
from sign_to_text import main as sign_to_text_main
from text_to_sign import main as text_to_sign_main

st.sidebar.title("🔄 Navigation")
page = st.sidebar.radio("Select Mode:", ["🤟 Sign to Text", "📖 Text to Sign"])

if page == "🤟 Sign to Text":
    sign_to_text_main()
elif page == "📖 Text to Sign":
    text_to_sign_main()
