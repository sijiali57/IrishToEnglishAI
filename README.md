# Irish to English Translation with Feedback

This project demonstrates how to build an AI model to translate Irish to English using the `MarianMT` model from Hugging Face. It also includes a simple web interface for users to provide feedback, which can be used to fine-tune the model further.

## Features
- Translate text from Irish to English using a pre-trained `MarianMT` model.
- Allow users to provide feedback on the translations.
- Collect user feedback for future fine-tuning of the model.

## Prerequisites
- Python 3.7 or higher
- Install the required libraries:
  ```bash
  pip install -r requirements.txt
  ```

## Usage
1. Run the web app:
   ```bash
   streamlit run app.py
   ```
2. Enter Irish text, translate it, and provide feedback in the web interface.

3. Fine-tune the model using collected feedback (instructions in the `fine_tune.py` file).

## File Structure
- `app.py`: Streamlit app for translation and feedback collection.
- `translate.py`: Core logic for text translation.
- `fine_tune.py`: Script for fine-tuning the MarianMT model using user feedback.
- `feedback_log.txt`: File to store user feedback.
- `requirements.txt`: Python dependencies.
- `README.md`: Project documentation.

## Contributing
Contributions are welcome! Feel free to submit issues or pull requests to improve the project.
