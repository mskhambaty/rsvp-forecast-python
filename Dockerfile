# Use official lightweight Python image
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Ensure the CSV sits in the correct spot
# If you're using another CSV name or path, adjust this step accordingly
# e.g. if you previously had data/rsvp_data.csv, you can symlink:
RUN ln -sf historical_rsvp_data.csv rsvp_data.csv

RUN python create_practical_model.py

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
