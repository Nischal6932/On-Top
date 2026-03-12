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
            import os
            import psutil
            model_path = os.path.join(os.path.dirname(__file__), "plant_disease_efficientnet.keras")
            print(f"🤖 Loading model from: {model_path}")
            
            # Check memory before loading
            memory_before = psutil.virtual_memory()
            print(f"💾 Memory before model: {memory_before.percent}% used")
            
            # Check if model file exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Get file size
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            print(f"📊 Model file size: {file_size:.1f} MB")
            
            model = tf.keras.models.load_model(model_path, compile=False)
            
            # Check memory after loading
            memory_after = psutil.virtual_memory()
            print(f"💾 Memory after model: {memory_after.percent}% used")
            print("✅ Model loaded successfully")
            
        except FileNotFoundError as e:
            print(f"❌ Model file error: {e}")
            raise e
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            # Don't raise exception, return None to allow fallback
            model = None
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



@app.route('/health', methods=['GET'])
def health():
    return {"status": "ok", "message": "Smart Farming AI is running"}, 200

@app.route('/test', methods=['GET'])
def test():
    return {"status": "ok", "message": "Test endpoint working", "model_loaded": model is not None}, 200

@app.route('/debug', methods=['GET'])
def debug_info():
    """Debug endpoint to check system status"""
    import os
    import sys
    
    debug_info = {
        'python_version': sys.version,
        'working_directory': os.getcwd(),
        'files_in_dir': os.listdir('.'),
        'model_files': [f for f in os.listdir('.') if f.endswith(('.keras', '.h5'))],
        'model_file_exists': os.path.exists('plant_disease_efficientnet.keras'),
        'model_file_size': None,
        'memory_info': None,
        'tensorflow_version': None,
        'numpy_version': None,
        'pillow_version': None
    }
    
    try:
        import tensorflow as tf
        debug_info['tensorflow_version'] = tf.__version__
    except:
        debug_info['tensorflow_version'] = 'Not available'
    
    try:
        import numpy as np
        debug_info['numpy_version'] = np.__version__
    except:
        debug_info['numpy_version'] = 'Not available'
    
    try:
        from PIL import Image
        debug_info['pillow_version'] = Image.__version__
    except:
        debug_info['pillow_version'] = 'Not available'
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        debug_info['memory_info'] = {
            'total_gb': round(memory.total / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'percent_used': memory.percent
        }
    except:
        debug_info['memory_info'] = 'psutil not available'
    
    if debug_info['model_file_exists']:
        try:
            size = os.path.getsize('plant_disease_efficientnet.keras') / (1024 * 1024)
            debug_info['model_file_size'] = f"{size:.1f} MB"
        except:
            debug_info['model_file_size'] = 'Could not determine size'
    
    return jsonify(debug_info)

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
    print(f"🔍 Request method: {request.method}")
    print(f"📁 Request files: {list(request.files.keys())}")
    print(f"📝 Request form: {dict(request.form)}")
    
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
        print("🔄 Processing POST request...")
        
        file = request.files.get('image')
        print(f"📷 File received: {file}, filename: {file.filename if file else 'None'}")

        # Prevent crashes if request is triggered without form data (health checks etc.)
        if request.method == "POST" and (file is None or file.filename == ""):
            print("❌ No file uploaded, returning error message")
            return render_template(
                "index.html",
                result=None,
                confidence=None,
                description="Please upload a plant leaf image.",
                treatment=None,
                soil_advice=None,
                irrigation_advice=None,
                weather_analysis=None,
                top2_predictions=None,
                ai_advice=None,
                chat_response=None
            )

        crop = request.form.get("crop")

        # Safety fallback if crop not selected
        if crop is None or crop.strip() == "":
            crop = "Unknown"

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

        # Safe moisture parsing
        try:
            moisture_val = int(moisture) if moisture is not None else 40
        except Exception:
            moisture_val = 40

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
            print("🖼️ Processing image...")
            img = Image.open(file).convert("RGB").resize((224,224))
            print(f"✅ Image processed successfully, shape: {img.size}")
        except Exception as e:
            print(f"❌ Image processing error: {e}")
            return render_template(
                "index.html",
                result=None,
                confidence=None,
                description=f"Image processing failed: {str(e)}. Please upload a valid leaf image.",
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
        print(f"🔢 Image array shape: {img.shape}")

        try:
            print("🧠 Loading model for prediction...")
            model = get_model()
            if model is None:
                print("❌ Model is None after get_model()")
                return render_template(
                    "index.html",
                    result=None,
                    confidence=None,
                    description="Model not available. Please check server configuration.",
                    treatment=None,
                    soil_advice=soil_advice,
                    irrigation_advice=irrigation_advice,
                    weather_analysis=weather_analysis,
                    top2_predictions=None,
                    ai_advice=None,
                    chat_response=None
                )
            
            print(f"🔮 Making prediction on image shape: {img.shape}")
            prediction = model.predict(img, verbose=0)
            print(f"📊 Raw prediction shape: {prediction.shape}")
            print(f"📈 Raw prediction values: {prediction}")
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return render_template(
                "index.html",
                result=None,
                confidence=None,
                description=f"Model prediction failed: {str(e)}. Please try again.",
                treatment=None,
                soil_advice=soil_advice,
                irrigation_advice=irrigation_advice,
                weather_analysis=weather_analysis,
                top2_predictions=None,
                ai_advice=None,
                chat_response=None
            )

        # --- Crop based class filtering ---
        if crop == "Pepper":
            allowed_classes = [i for i, c in enumerate(class_names) if "Pepper" in c]
        elif crop == "Potato":
            allowed_classes = [i for i, c in enumerate(class_names) if "Potato" in c]
        elif crop == "Tomato":
            allowed_classes = [i for i, c in enumerate(class_names) if "Tomato" in c]
        else:
            # fallback to all classes if crop is unknown
            allowed_classes = list(range(len(class_names)))

        # Safety check to avoid empty filtering
        if len(allowed_classes) == 0:
            allowed_classes = list(range(len(class_names)))

        # Filter predictions to only allowed crop classes
        preds = np.squeeze(prediction)
        filtered_predictions = np.array(preds)[allowed_classes]

        # Prevent crash if something goes wrong with filtering
        if len(filtered_predictions) == 0:
            filtered_predictions = np.array(preds)
            allowed_classes = list(range(len(class_names)))

        # Get sorted indices (highest to lowest)
        sorted_idx = np.argsort(filtered_predictions)[::-1]
        print(f"📊 Sorted indices: {sorted_idx}")
        
        # Safety check
        if len(sorted_idx) == 0:
            print("❌ No sorted indices available")
            return render_template(
                "index.html",
                result=None,
                confidence=None,
                description="Prediction could not be generated. Please upload a clearer image.",
                treatment=None,
                soil_advice=soil_advice,
                irrigation_advice=irrigation_advice,
                weather_analysis=weather_analysis,
                top2_predictions=None,
                ai_advice=None,
                chat_response=None
            )

        # Get best and second best predictions
        best_idx_local = sorted_idx[0]
        second_idx_local = sorted_idx[1] if len(sorted_idx) > 1 else sorted_idx[0]

        # Convert to original class indices
        best_idx = allowed_classes[best_idx_local]
        second_idx = allowed_classes[second_idx_local]

        top2_predictions = [
            (class_names[best_idx], float(filtered_predictions[best_idx_local])),
            (class_names[second_idx], float(filtered_predictions[second_idx_local]))
        ]

        # Confidence based only on the filtered crop classes
        confidence = float(filtered_predictions[best_idx_local])
        second_confidence = float(filtered_predictions[second_idx_local])

        print(f"🎯 Prediction result: {class_names[best_idx]} with confidence {confidence}")
        print(f"📈 Confidence values: best={confidence}, second={second_confidence}")

        # Confidence threshold to avoid false disease alarms
        if confidence < 0.7:
            result = "Leaf appears healthy or disease is unclear"
            description = "The model confidence is low. The leaf likely appears healthy or symptoms are not clear."
            treatment = "Monitor the plant and upload a clearer image if symptoms develop."
            print("🟢 Low confidence - marking as healthy/unclear")
        else:
            result = class_names[best_idx]
            print(f"🔴 High confidence - disease detected: {result}")

            # Skip LLM if plant is healthy
            if "healthy" in result.lower():
                description = "The plant appears healthy with no visible disease symptoms."
                treatment = "Continue regular irrigation, monitor plant health, and maintain good soil nutrition."
                ai_advice = None
                print("🟢 Plant is healthy")
            else:
                # Do not call LLM here so page loads faster.
                # The frontend can request detailed AI advice using the /ai_advice API.
                description = "Disease detected. Detailed AI advice will load shortly."
                treatment = None
                ai_advice = None
                print("🔴 Disease detected - AI advice available")

        confidence = round(confidence * 100, 2)
        print(f"📊 Final confidence: {confidence}%")
        print("✅ Prediction processing complete")
        
    else:
        print("📄 GET request - showing upload form")

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
    import os
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print("=" * 50)
    print("🌿 Smart Farming AI Starting...")
    print(f"Port: {port}")
    print(f"Debug: {debug_mode}")
    print(f"Environment: {os.environ.get('ENVIRONMENT', 'development')}")
    print("=" * 50)
    
    # Test model loading
    try:
        test_model = get_model()
        print(f"✅ Model loaded: {test_model is not None}")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
