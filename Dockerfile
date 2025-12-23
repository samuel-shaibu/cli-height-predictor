FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy requirements first 
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Source code AND the trained model
COPY src/ ./src/
COPY model.joblib .

# Set the entry point
# When the user runs the container, this command fires automatically
ENTRYPOINT [ "python", "src/app.py" ]