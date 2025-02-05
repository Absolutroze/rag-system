# Step 1: Use an official Python runtime as a parent image
FROM python:3.9-slim

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy the current directory contents into the container
COPY . /app

# Step 4: Install the dependencies
RUN pip install fastapi uvicorn langchain langchain_ollama fastembed
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Step 5: Expose port 80 for FastAPI
EXPOSE 80

# Step 6: Define environment variable (optional)
ENV PYTHONUNBUFFERED 1

# Step 7: Run FastAPI when the container launches
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
