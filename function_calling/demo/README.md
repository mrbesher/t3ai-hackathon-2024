# Function Calling Chatbot
## Setup
1. Install the required packages using the following command:
```
python3 -m venv func-tool
source func-tool/bin/activate
pip install -U pip
pip install -r requirements.txt
```

# Run the model using the CLI:
```
python3 main.py
```

# Run the model using the GUI:
```
streamlit run app.py --server.port 5051 --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false
```