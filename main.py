import os
from flask import Flask, request, jsonify
import google.generativeai as genai
import requests
from urllib.parse import urlparse

# Configuration de l'API Google AI
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

app = Flask(__name__)

# Configuration du modèle
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
)

# Dictionnaire global pour stocker l'historique de chaque utilisateur
user_histories = {}

def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    try:
        file = genai.upload_file(path, mime_type=mime_type)
        print(f"Uploaded file '{file.display_name}' as: {file.uri}")
        return file
    except Exception as e:
        print(f"Error uploading file: {e}")
        return None

@app.route('/api/gemini_vision', methods=['GET'])
def gemini_vision_get():
    user_id = request.args.get('user_id')
    text = request.args.get('text')
    image_url = request.args.get('image_url')

    if not user_id:
        return jsonify({'error': 'user_id parameter not provided'}), 400

    # Récupérer l'historique de l'utilisateur ou en créer un nouveau
    if user_id not in user_histories:
        user_histories[user_id] = []

    user_history = user_histories[user_id]

    try:
        if image_url and text:
            image_path = download_image(image_url)
            if image_path:
                uploaded_file = upload_to_gemini(image_path, mime_type="image/jpeg")
                if uploaded_file:
                    user_history.append({
                        "role": "user",
                        "parts": [uploaded_file.uri, text],
                    })
                else:
                    return jsonify({'error': 'Failed to upload image'}), 500
            else:
                return jsonify({'error': 'Failed to download image'}), 500

        elif text:
            user_history.append({
                "role": "user",
                "parts": [text],
            })

        chat_session = model.start_chat(history=user_history)
        response = chat_session.send_message(text)
        user_history.append({"role": "model", "parts": [response.text]})
        return jsonify({'response': response.text})

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({'error': 'An error occurred while processing your request'}), 500

def download_image(image_url):
    """Télécharge une image depuis une URL et retourne le chemin local."""
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        image_path = os.path.join("/tmp", os.path.basename(urlparse(image_url).path))
        with open(image_path, 'wb') as out_file:
            out_file.write(response.content)
        return image_path
    except requests.RequestException as e:
        print(f"Error downloading image: {e}")
        return None

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
