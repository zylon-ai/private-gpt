import streamlit as st
import ingest
import secureGPT

def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Home", "Ingest", "SecureGPT"])

    if selection == "Home":
        st.title("Home")
        st.write("Welcome to the home page!")
    elif selection == "Ingest":
        st.title("Ingest")
        ingest.run()  # assuming your ingest.py has a run function
    elif selection == "SecureGPT":
        st.title("SecureGPT")
        secureGPT.run()  # assuming your secureGPT.py has a run function

if __name__ == "__main__":
    main()
    