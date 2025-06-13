import os
import io
import base64
import requests
from flask import Flask, render_template, request

app = Flask(__name__)

# URL de l'API de prédiction sur Heroku
API_BASE = "https://img-seg-api-f284f7d46372.herokuapp.com/" # "http://127.0.0.1:5000/"

# Dossier des images réelles
IMAGES_FOLDER = "static/images"

@app.route("/", methods=["GET", "POST"])
def index():
    image_id = None
    real_image_url = None
    real_mask_url = None
    predicted_mask = None
    error_message = None
    processing = False

    # Obtenir la liste des images disponibles
    try:
        images = [
            f.replace(".png", "") for f in os.listdir(IMAGES_FOLDER)
            if f.endswith(".png")
        ]
    except Exception as e:
        error_message = f"Erreur récupération images locales: {e}"
        images = []

    # Si l'utilisateur soumet le formulaire
    if request.method == "POST":
        processing = True  
        custom_image = request.files.get("custom_image")
        image_id = request.form.get("image_id")

        try:
            # Cas 1 : image personnalisée
            if custom_image and custom_image.filename != "":
                image_bytes = custom_image.read()
                files = {"image": (custom_image.filename, io.BytesIO(image_bytes), "image/png")}
                pred = requests.post(f"{API_BASE}/predict", files=files)

                if pred.status_code == 200:
                    img_base64 = base64.b64encode(pred.content).decode('utf-8')
                    predicted_mask = f"data:image/png;base64,{img_base64}"
                else:
                    error_message = f"Erreur prédiction (code {pred.status_code})"

            # Cas 2 : image du dataset
            else:
                real_image_url = f"/static/images/{image_id}.png"
                mask_id = image_id.replace("_leftImg8bit", "_gtFine_labelIds")
                real_mask_url = f"/static/masks/{mask_id}.png"

                with open(os.path.join(IMAGES_FOLDER, f"{image_id}.png"), "rb") as img_file:
                    image_bytes = img_file.read()

                files = {"image": ("image.png", io.BytesIO(image_bytes), "image/png")}
                pred = requests.post(f"{API_BASE}/predict", files=files)

                if pred.status_code == 200:
                    img_base64 = base64.b64encode(pred.content).decode('utf-8')
                    predicted_mask = f"data:image/png;base64,{img_base64}"
                else:
                    error_message = f"Erreur prédiction (code {pred.status_code})"

        except Exception as e:
            error_message = f"Erreur lors de la prédiction: {e}"

        processing = False

    # Affichage de la page
    return render_template(
        "index.html",
        images=images,
        selected_id=image_id,
        real_image_url=real_image_url,
        real_mask_url=real_mask_url,
        predicted_mask_url=predicted_mask,
        error_message=error_message,
        processing=processing
    )

if __name__ == "__main__":
    app.run(debug=True, port=5001)
