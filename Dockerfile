FROM python:3.11-slim

RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY prediction.py best_plant_diagnosis.keras gate_model.keras class_names.npy ./

# The app writes feedback images, the retrain marker, and retrained model
# weights into /app, so the runtime user needs ownership of it.
RUN mkdir -p /app/feedback_data && chown -R user:user /app

USER user
ENV HOME=/home/user

EXPOSE 7860
CMD ["uvicorn", "prediction:app", "--host", "0.0.0.0", "--port", "7860"]
