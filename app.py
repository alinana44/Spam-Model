import streamlit as st
from spam_classifier_model import classify_new_email  # Import the classify function from your model script

# Streamlit UI
st.title("Email Spam Classifier")
st.write("This is a simple email spam classifier built using Naive Bayes.")

# Create a text area for users to input their email
new_email = st.text_area("Enter the email content:")

# Button to trigger classification
if st.button("Classify"):
    if new_email:
        # Classify the email
        prediction = classify_new_email(new_email)
        st.write(f"The email is classified as: {prediction}")
    else:
        st.warning("Please enter an email to classify.")
