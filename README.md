# Disease Prediction Web App
## Project Description
This project uses a Random Forest Classifier trained on health-related data to classify if a person is at risk of developing diabetes or heart disease. Users can input their health parameters via a web form, and the model returns a predicted diagnosis.
### 🧠 Technologies Used

Python 3.9.13

Flask

NumPy

Pandas

Scikit-learn

HTML, CSS, JS

Pickle for saving ML models

Gunicorn for production WSGI server


### 🚀 How to Run Locally

# Clone the repository
git clone https://github.com/Swarup899/render_healthdata.git
cd render_healthdata

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
