FROM python:3.10

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install Flask pymongo mlflow pandas numpy

COPY . .

EXPOSE 5000

CMD ["python3", "app.py"]