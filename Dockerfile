FROM python:3.8

# Install system dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx



EXPOSE 8080
WORKDIR /app

COPY . ./

RUN pip install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
