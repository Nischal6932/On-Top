from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("plant_disease_efficientnet.keras")

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
disease_info = {

"Tomato_Late_blight": {
"description": "Late blight is a fungal disease that causes dark lesions on leaves and stems.",
"treatment": "Remove infected leaves, avoid overhead watering, and apply copper-based fungicides."
},

"Tomato_Early_blight": {
"description": "Early blight causes brown spots with concentric rings on tomato leaves.",
"treatment": "Use disease-resistant varieties and apply fungicides when symptoms appear."
},

"Tomato_Leaf_Mold": {
"description": "Leaf mold appears as yellow spots on the upper leaf surface and mold underneath.",
"treatment": "Improve air circulation and apply appropriate fungicides."
},

"Tomato_healthy": {
"description": "The plant appears healthy with no visible disease symptoms.",
"treatment": "Continue regular irrigation and monitoring."
},

"Potato_Early_blight": {
"description": "Early blight causes brown lesions with target-like rings on potato leaves.",
"treatment": "Use crop rotation and fungicide sprays."
},

"Potato_Late_blight": {
"description": "Late blight causes water-soaked lesions that quickly turn brown.",
"treatment": "Remove infected plants and apply protective fungicides."
}

}

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

    if request.method == "POST":

        file = request.files['image']
        crop = request.form.get("crop")
        soil = request.form.get("soil")
        moisture = request.form.get("moisture")
        weather = request.form.get("weather")

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

        img = Image.open(file).convert("RGB").resize((224,224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)

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

        best_idx_local = top2_local[0]
        second_idx_local = top2_local[1] if len(top2_local) > 1 else top2_local[0]

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

            if result in disease_info:
                description = disease_info[result]["description"]
                treatment = disease_info[result]["treatment"]
            else:
                description = "No detailed description available."
                treatment = "Consult a local agricultural expert."

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
        top2_predictions=top2_predictions
    )


if __name__ == "__main__":
    app.run(debug=True)