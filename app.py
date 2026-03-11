from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from ollama_client import ask_llm

app = Flask(__name__)

model = None


def get_model():
    global model
    if model is None:
        try:
            model = tf.keras.models.load_model("plant_disease_model.h5", compile=False)
        except Exception as e:
            print("Model loading failed:", e)
            raise e
    return model

# Disease classes
class_names = [
"Pepper__bell___Bacterial_spot",
"Pepper__bell___healthy",
"Potato___Early_blight",
"Potato___Late_blight",
"Potato___healthy",
"Tomato_Bacterial_spot",
"Tomato_Early_blight",
"Tomato_Late_blight",
"Tomato_Leaf_Mold",
"Tomato_Septoria_leaf_spot",
"Tomato_Spider_mites",
"Tomato_Target_Spot",
"Tomato_Yellow_Leaf_Curl_Virus",
"Tomato_mosaic_virus",
"Tomato_healthy"
]



@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200

@app.route("/ai_advice", methods=["POST"])
def ai_advice_endpoint():

    data = request.json

    crop = data.get("crop")
    disease = data.get("disease")
    soil = data.get("soil")
    moisture = data.get("moisture")
    weather = data.get("weather")
    question = data.get("question")

    # Build prompt depending on whether farmer asked a question
    if question and question.strip() != "":
        prompt = f"""
You are an expert agricultural advisor helping farmers in India.

The farmer asked a specific question about their crop.

Farmer Question:
{question}

Crop: {crop}

Answer ONLY the farmer's question clearly and directly.
Do NOT explain the detected disease unless the question asks about it.
Use short bullet points and simple farmer‑friendly language.
"""
    else:
        prompt = f"""
You are an expert agricultural advisor helping farmers in India.

Crop: {crop}
Detected Disease: {disease}
Soil Type: {soil}
Soil Moisture: {moisture}%
Weather: {weather}

Explain the disease clearly.

Provide:
• What the disease is
• Why it occurs
• Treatment steps
• Prevention tips

Use simple bullet points suitable for farmers.
"""

    try:
        response = ask_llm(prompt).strip()
        return jsonify({"advice": response})
    except Exception:
        return jsonify({"advice": "AI advice service unavailable"})

@app.route('/', methods=['GET', 'POST'])
def predict():

    result = None
    confidence = None
    description = None
    treatment = None
    soil_advice = None
    irrigation_advice = None
    weather_analysis = None
    top2_predictions = None
    ai_advice = None
    chat_response = None

    if request.method == "POST":

        file = request.files.get('image')
        if file is None or file.filename == "":
            return render_template(
                "index.html",
                result=None,
                confidence=None,
                description="No image uploaded. Please upload a leaf image.",
                treatment=None,
                soil_advice=None,
                irrigation_advice=None,
                weather_analysis=None,
                top2_predictions=None,
                ai_advice=None,
                chat_response=None
            )
        crop = request.form.get("crop")
        soil = request.form.get("soil")
        moisture = request.form.get("moisture")
        weather = request.form.get("weather")
        user_question = request.form.get("question")

        # --- Environment Analysis (rule-based) ---

        # Soil compatibility check
        if crop == "Rice" and soil == "Clay":
            soil_advice = "Good soil choice. Clay soil retains water well and is suitable for rice cultivation."
        elif crop == "Tomato" and soil == "Loam":
            soil_advice = "Loamy soil is ideal for tomato plants due to good drainage and nutrient balance."
        elif crop == "Potato" and soil == "Sandy":
            soil_advice = "Sandy soil supports good potato tuber development and drainage."
        else:
            soil_advice = f"{soil} soil can grow {crop}, but monitoring nutrients and drainage is recommended."

        # Moisture based irrigation advice
        moisture_val = int(moisture) if moisture else 40

        if moisture_val < 30:
            irrigation_advice = "Soil moisture is low. Increase irrigation frequency."
        elif 30 <= moisture_val <= 70:
            irrigation_advice = "Soil moisture is in optimal range. Maintain current watering schedule."
        else:
            irrigation_advice = "Soil moisture is high. Reduce irrigation to avoid root diseases."

        # Weather risk analysis
        if weather == "Humid":
            weather_analysis = "Humid conditions may increase fungal disease risk. Monitor leaves closely."
        elif weather == "Rainy":
            weather_analysis = "Rainy weather can spread plant pathogens quickly. Ensure good drainage."
        elif weather == "Hot":
            weather_analysis = "High temperatures may stress plants. Maintain adequate irrigation."
        else:
            weather_analysis = "Weather conditions appear stable for crop growth."

        try:
            img = Image.open(file).convert("RGB").resize((224,224))
        except Exception:
            return render_template(
                "index.html",
                result=None,
                confidence=None,
                description="Invalid image file. Please upload a valid leaf image.",
                treatment=None,
                soil_advice=None,
                irrigation_advice=None,
                weather_analysis=None,
                top2_predictions=None,
                ai_advice=None,
                chat_response=None
            )
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        model = get_model()
        prediction = model.predict(img, verbose=0)

        # --- Crop based class filtering ---
        if crop == "Pepper":
            allowed_classes = [i for i,c in enumerate(class_names) if "Pepper" in c]
        elif crop == "Potato":
            allowed_classes = [i for i,c in enumerate(class_names) if "Potato" in c]
        elif crop == "Tomato":
            allowed_classes = [i for i,c in enumerate(class_names) if "Tomato" in c]
        else:
            allowed_classes = list(range(len(class_names)))

        # Filter predictions to only allowed crop classes
        filtered_predictions = prediction[0][allowed_classes]

        # Get Top-2 predictions among allowed classes
        top2_local = np.argsort(filtered_predictions)[-2:][::-1]

        best_idx_local = int(top2_local[0])
        second_idx_local = int(top2_local[1]) if len(top2_local) > 1 else int(top2_local[0])

        best_idx = allowed_classes[best_idx_local]
        second_idx = allowed_classes[second_idx_local]

        top2_predictions = [
            (class_names[best_idx], float(filtered_predictions[best_idx_local])),
            (class_names[second_idx], float(filtered_predictions[second_idx_local]))
        ]

        # Confidence based only on the filtered crop classes
        confidence = float(filtered_predictions[best_idx_local])
        second_confidence = float(filtered_predictions[second_idx_local])

        # Confidence threshold to avoid false disease alarms
        if confidence < 0.7:
            result = "Leaf appears healthy or disease is unclear"
            description = "The model confidence is low. The leaf likely appears healthy or symptoms are not clear."
            treatment = "Monitor the plant and upload a clearer image if symptoms develop."
        else:
            result = class_names[best_idx]

            # Skip LLM if the plant is healthy
            if "healthy" in result.lower():
                description = "The plant appears healthy with no visible disease symptoms."
                treatment = "Continue regular irrigation, monitor plant health, and maintain good soil nutrition."
                ai_advice = None
            else:
                # Do not call the LLM here so the page loads faster.
                # The frontend can request detailed AI advice using the /ai_advice API.
                description = "Disease detected. Detailed AI advice will load shortly."
                treatment = None
                ai_advice = None

        confidence = round(confidence * 100, 2)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        description=description,
        treatment=treatment,
        soil_advice=soil_advice,
        irrigation_advice=irrigation_advice,
        weather_analysis=weather_analysis,
        top2_predictions=top2_predictions,
        ai_advice=ai_advice,
        chat_response=chat_response
    )


if __name__ == "__main__":
    app.run(debug=True)
