import streamlit as st
from translate import translate_text

# Path to the feedback log
FEEDBACK_LOG_PATH = "feedback_log.txt"

def save_feedback(irish_text, translated_text, user_feedback):
    """
    Save user feedback to feedback_log.txt.

    Args:
        irish_text (str): The original Irish text.
        translated_text (str): The translation provided by the model.
        user_feedback (str): The user's feedback (corrected translation or comments).
    """
    with open(FEEDBACK_LOG_PATH, "a") as file:
        file.write(f"Irish: {irish_text}\n")
        file.write(f"Translated: {translated_text}\n")
        file.write(f"Feedback: {user_feedback}\n")
        file.write("-----\n")

# Streamlit UI
st.title("Irish to English Translation")
st.write("Enter Irish text below to translate it into English.")

# User input for translation
irish_text = st.text_area("Input Irish Text", key="irish_text")

if st.button("Translate", key="translate_button"):
    if irish_text.strip():
        # Perform translation
        translated_text = translate_text(irish_text)
        st.session_state.translated_text = translated_text
        st.session_state.irish_text = irish_text
    else:
        st.warning("Please enter text to translate.")

# Display translation if it exists
if "translated_text" in st.session_state:
    st.write("### Translated Text:")
    st.write(st.session_state.translated_text)

    # User input for feedback
    st.write("### Provide Feedback")
    user_feedback = st.text_area("Suggest a better translation or leave comments.", key="user_feedback")

    if st.button("Submit Feedback", key="submit_feedback_button"):
        if user_feedback.strip():
            save_feedback(
                st.session_state.irish_text,
                st.session_state.translated_text,
                user_feedback
            )
            st.success("Thank you! Your feedback has been saved.")
            # Clear session state for feedback after submission
            st.session_state.pop("translated_text", None)
            st.session_state.pop("irish_text", None)
        else:
            st.warning("Please provide feedback before submitting.")