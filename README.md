# Final Project - Intro to Data

This is the final project for the Intro to Data Science course with Professor Zhang. The project involves an analytical approach to the optimal recommender system culminating in the a hybrid recommender chatbot using Streamlit.

## Requirements

To run this project, you need to have the following installed:

- Python 3.x
- Streamlit

## Installation

To install Streamlit and dependencies, use the following command:

```
pip install streamlit pandas numpy tensorflow scikit-learn
```

If you have ML and data dependencies installed just install Streamlit.

## Running the Project

To run the chatbot, use the following command:

```
streamlit run chatbot_streamlit.py
```

## Project Structure

- `chatbot_streamlit.py`: The main script to run the Streamlit chatbot. All other code (EDA, metrics, etc) is in `project.ipynb`.
- `README.md`: This file.

## Usage

Follow the instructions above to install the required packages and run the chatbot application.

The chatbot takes in a user id from the MovieLens dataset. You can ask the chatbot to recommend movies to the selected user. Functionality for outputting disrecommended movies is supported. Additionally, a content-based recommender is included in the design: ask the chatbot to recommend movies to the user based on the input watched title. 

## License

This project is licensed under the Nile License.

