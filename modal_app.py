"""Deploy the Plant Diagnosis FastAPI engine on Modal (modal.com).

Usage:
    pip install modal
    modal setup            # one-time login (GitHub account, free Starter plan)
    modal deploy modal_app.py

Modal builds a container image with TensorFlow + the model files, and serves
the existing FastAPI app from prediction.py unchanged at a public
https://...modal.run URL. Containers scale to zero when idle, so free
credits are only consumed while requests are being handled.
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .add_local_file("prediction.py", "/root/prediction.py")
    .add_local_file("best_plant_diagnosis.keras", "/root/best_plant_diagnosis.keras")
    .add_local_file("gate_model.keras", "/root/gate_model.keras")
    .add_local_file("class_names.npy", "/root/class_names.npy")
)

app = modal.App("plant-diagnosis", image=image)


@app.function(memory=4096, timeout=600)
@modal.asgi_app()
def fastapi_app():
    # Importing prediction loads both Keras models (takes ~30-60 s on a cold
    # start); after that the container stays warm for subsequent requests.
    import prediction

    return prediction.app
