FROM python:3.12-slim

WORKDIR /app

COPY ./requirements-docker.txt ./

RUN pip install -r requirements-docker.txt

COPY ./app.py ./app.py
COPY ./data/external/plot_data.csv ./data/external/plot_data.csv
COPY ./data/processed/test_data.csv ./data/processed/test_data.csv
COPY ./models/ ./models/

EXPOSE 8000

CMD ["streamlit","run","app.py","--server.port","8000","--server.address","0.0.0.0"]
