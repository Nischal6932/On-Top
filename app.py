from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from ollama_client import ask_llm
from gtts import gTTS
import os
from deep_translator import GoogleTranslator


app = Flask(__name__)

# Translations for multi-language support
translations = {
    "English": {
        "title": "Smart Farming Assistant",
        "upload": "Upload Leaf Image",
        "crop": "Select Crop",
        "soil": "Soil Type",
        "moisture": "Soil Moisture Level",
        "weather": "Current Weather",
        "question": "Ask AI about your crop (optional)",
        "button": "🔍 Analyze My Crop",
        "disease_desc": "Disease Description",
        "ai_guidance": "AI Disease Guidance",
        "ai_answer": "AI Answer to Your Question",
        "treatment": "Recommended Treatment",
        "soil_card": "Soil Compatibility",
        "irrigation": "Irrigation Advice",
        "weather_card": "Weather Analysis",
        "alternatives": "Alternative Possibilities"
        ,"crop_tomato": "Tomato",
        "crop_potato": "Potato",
        "crop_pepper": "Pepper",
        "soil_clay": "Clay",
        "soil_loam": "Loam",
        "soil_sandy": "Sandy",
        "soil_silt": "Silt",
        "weather_dry": "Dry",
        "weather_humid": "Humid",
        "weather_rainy": "Rainy",
        "weather_hot": "Hot"
    },
    "Hindi": {
        "title": "स्मार्ट खेती सहायक",
        "upload": "पत्ती की तस्वीर अपलोड करें",
        "crop": "फसल चुनें",
        "soil": "मिट्टी का प्रकार",
        "moisture": "मिट्टी की नमी",
        "weather": "मौसम",
        "question": "अपने फसल के बारे में पूछें",
        "button": "🔍 विश्लेषण करें",
        "disease_desc": "रोग विवरण",
        "ai_guidance": "AI रोग मार्गदर्शन",
        "ai_answer": "आपके प्रश्न का उत्तर",
        "treatment": "उपचार सुझाव",
        "soil_card": "मिट्टी अनुकूलता",
        "irrigation": "सिंचाई सलाह",
        "weather_card": "मौसम विश्लेषण",
        "alternatives": "वैकल्पिक संभावनाएँ"
        ,"crop_tomato": "टमाटर",
        "crop_potato": "आलू",
        "crop_pepper": "मिर्च",
        "soil_clay": "चिकनी मिट्टी",
        "soil_loam": "दोमट",
        "soil_sandy": "रेतीली",
        "soil_silt": "गाद",
        "weather_dry": "सूखा",
        "weather_humid": "नमी",
        "weather_rainy": "बरसात",
        "weather_hot": "गर्म"
    },
    "Telugu": {
        "title": "స్మార్ట్ వ్యవసాయ సహాయకుడు",
        "upload": "ఆకు చిత్రం అప్లోడ్ చేయండి",
        "crop": "పంటను ఎంచుకోండి",
        "soil": "మట్టి రకం",
        "moisture": "మట్టి తేమ",
        "weather": "వాతావరణం",
        "question": "మీ పంట గురించి అడగండి",
        "button": "🔍 విశ్లేషించండి",
        "disease_desc": "రోగ వివరణ",
        "ai_guidance": "AI వ్యాధి సూచనలు",
        "ai_answer": "మీ ప్రశ్నకు సమాధానం",
        "treatment": "చికిత్స సూచనలు",
        "soil_card": "మట్టి అనుకూలత",
        "irrigation": "పారుదల సూచనలు",
        "weather_card": "వాతావరణ విశ్లేషణ",
        "alternatives": "ప్రత్యామ్నాయ అవకాశాలు"
        ,"crop_tomato": "టమోటా",
        "crop_potato": "బంగాళాదుంప",
        "crop_pepper": "మిర్చి",
        "soil_clay": "మట్టి",
        "soil_loam": "లోమ్",
        "soil_sandy": "ఇసుక",
        "soil_silt": "సిల్ట్",
        "weather_dry": "ఎండ",
        "weather_humid": "తేమ",
        "weather_rainy": "వర్షం",
        "weather_hot": "వేడి"
    }
}

# Configure file upload settings
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.jpeg', '.png', '.webp']
app.config['UPLOAD_FOLDER'] = '/tmp'  # Use temp directory for uploads

model = None


def get_model():
    """
    Memory-efficient model loading with fallback for low-memory environments
    """
    global model
    if model is None:
        try:
            import os
            import gc
            
            # Check available memory first
            try:
                import psutil
                memory = psutil.virtual_memory()
                available_mb = memory.available / (1024 * 1024)
                print(f"💾 Available memory: {available_mb:.1f} MB")
                
                # If less than 200MB available, use fallback mode
                if available_mb < 200:
                    print("⚠️ Low memory detected, using fallback mode")
                    model = None
                    return model
                    
            except ImportError:
                print("💾 Memory monitoring not available")
            
            # Try multiple possible model paths for deployment compatibility
            possible_paths = [
                "plant_disease_efficientnet.keras",
                os.path.join(os.path.dirname(__file__), "plant_disease_efficientnet.keras"),
                os.path.join(os.getcwd(), "plant_disease_efficientnet.keras"),
                "/app/plant_disease_efficientnet.keras"
            ]
            
            model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path is None:
                print("❌ Model file not found, using fallback mode")
                model = None
                return model
            
            print(f"🤖 Loading model from: {model_path}")
            
            # Get file size
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            print(f"📊 Model file size: {file_size:.1f} MB")
            
            # Load model with memory-efficient settings
            try:
                # Try loading with reduced memory footprint
                model = tf.keras.models.load_model(model_path, compile=False)
                
                # Force garbage collection
                gc.collect()
                
                print("✅ Model loaded successfully")
                
            except Exception as e:
                print(f"❌ Model loading failed: {e}")
                print("🔄 Using fallback mode")
                model = None
                gc.collect()
            
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            model = None
    
    return model

def get_fallback_prediction(img_array):
    """
    Fallback prediction using simple image analysis when model is not available
    """
    try:
        import numpy as np
        
        # Simple image analysis based on color distribution
        # This is a very basic fallback - in production you'd want a better solution
        
        # Calculate average color values
        avg_red = np.mean(img_array[:, :, 0])
        avg_green = np.mean(img_array[:, :, 1])
        avg_blue = np.mean(img_array[:, :, 2])
        
        # Calculate green ratio (indicator of plant health)
        total = avg_red + avg_green + avg_blue
        green_ratio = avg_green / total if total > 0 else 0
        
        # Simple heuristic based on color analysis
        if green_ratio > 0.4:
            # High green content - likely healthy
            return "healthy", 0.75
        elif green_ratio > 0.3:
            # Moderate green content - some issues
            return "moderate_risk", 0.60
        else:
            # Low green content - potential disease
            return "disease_risk", 0.55
            
    except Exception as e:
        print(f"Fallback prediction error: {e}")
        return "unknown", 0.50

def validate_upload_file(file):
    """Validate uploaded file for security and compatibility"""
    if not file or file.filename == '':
        return False, "No file selected"
    
    # Check file extension
    filename = file.filename.lower()
    allowed_extensions = app.config['UPLOAD_EXTENSIONS']
    if not any(filename.endswith(ext) for ext in allowed_extensions):
        return False, f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
    
    # Check file size (additional validation)
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset to beginning
    
    max_size = app.config['MAX_CONTENT_LENGTH']
    if file_size > max_size:
        return False, f"File too large. Maximum size: {max_size // (1024*1024)}MB"
    
    return True, "File valid"

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



@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    language = "English"
    t = translations.get(language, translations["English"])
    return render_template(
        "index.html",
        t=t,
        language=language,
        result=None,
        confidence=None,
        description=f"File too large. Maximum size is {app.config['MAX_CONTENT_LENGTH'] // (1024*1024)}MB. Please upload a smaller image.",
        treatment=None,
        soil_advice=None,
        irrigation_advice=None,
        weather_analysis=None,
        top2_predictions=None,
        ai_advice=None,
        chat_response=None
    ), 413

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
    language = data.get("language", "English")

    # Build prompt depending on whether farmer asked a question
    if question and question.strip() != "":
        prompt = f"""
You are an expert agricultural advisor helping farmers in India.

Respond STRICTLY in {language} language.
DO NOT use English at all.
If needed, translate your full answer into {language}.
Use simple, clear, farmer-friendly words in {language}.

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

Respond STRICTLY in {language} language.
DO NOT use English at all.
If needed, translate your full answer into {language}.
Use simple, clear, farmer-friendly words in {language}.

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
        print(f"🌍 Selected language: {language}")
        response = ask_llm(prompt).strip()

        # Force translation using Google Translator (reliable)
        try:
            if language != "English":
                lang_map = {
                    "English": "en",
                    "Hindi": "hi",
                    "Telugu": "te",
                    "Tamil": "ta",
                    "Kannada": "kn"
                }

                target_lang = lang_map.get(language, "en")

                translated_response = GoogleTranslator(source='auto', target=target_lang).translate(response)
                if translated_response:
                    response = translated_response

        except Exception as e:
            print(f"Translation failed: {e}")

        # Generate voice output using gTTS
        try:
            lang_map = {
                "English": "en",
                "Hindi": "hi",
                "Telugu": "te",
                "Tamil": "ta",
                "Kannada": "kn"
            }

            tts_lang = lang_map.get(language, "en")
            audio_path = os.path.join("static", "output.mp3")

            tts = gTTS(text=response, lang=tts_lang)
            tts.save(audio_path)

            audio_url = "/static/output.mp3"

        except Exception as e:
            print(f"Audio generation failed: {e}")
            audio_url = None

        return jsonify({
            "advice": response,
            "audio": audio_url
        })
    except Exception:
        return jsonify({
            "advice": "AI advice service unavailable",
            "audio": None
        })

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
    language = request.form.get("language", "English") if request.method == "POST" else "English"
    t = translations.get(language, translations["English"])

    if request.method == "POST":
        print("🔄 Processing POST request...")
        
        file = request.files.get('image')
        print(f"📷 File received: {file}, filename: {file.filename if file else 'None'}")

        # Validate uploaded file
        if file is None or file.filename == "":
            print("❌ No file uploaded, returning error message")
            return render_template(
                "index.html",
                t=t,
                language=language,
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
        
        # Validate file before processing
        is_valid, validation_message = validate_upload_file(file)
        if not is_valid:
            print(f"❌ File validation failed: {validation_message}")
            return render_template(
                "index.html",
                t=t,
                language=language,
                result=None,
                confidence=None,
                description=f"Invalid file: {validation_message}",
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
            file.seek(0)
            img = Image.open(file.stream).convert("RGB").resize((224, 224))
            print(f"✅ Image processed successfully, shape: {img.size}")
            
        except Exception as e:
            print(f"❌ Image processing error: {e}")
            error_msg = str(e)
            if "cannot identify image file" in error_msg.lower():
                error_msg = "Invalid or corrupted image file. Please upload a valid image."
            elif "image file is truncated" in error_msg.lower():
                error_msg = "Image file is corrupted or incomplete. Please try a different image."
            
            return render_template(
                "index.html",
                t=t,
                language=language,
                result=None,
                confidence=None,
                description=f"Image processing failed: {error_msg}",
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
                    t=t,
                    language=language,
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
                t=t,
                language=language,
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
                t=t,
                language=language,
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

    # 🔥 FORCE TRANSLATION FOR ALL OUTPUTS
    def translate_text(text):
        try:
            if language != "English" and text:
                lang_map = {
                    "English": "en",
                    "Hindi": "hi",
                    "Telugu": "te",
                    "Tamil": "ta",
                    "Kannada": "kn"
                }
                return GoogleTranslator(source='auto', target=lang_map.get(language, "en")).translate(text)
        except Exception as e:
            print(f"Translation error: {e}")
        return text

    # Apply translation to outputs
    description = translate_text(description)
    treatment = translate_text(treatment)
    soil_advice = translate_text(soil_advice)
    irrigation_advice = translate_text(irrigation_advice)
    weather_analysis = translate_text(weather_analysis)
    result = translate_text(result)

    return render_template(
        "index.html",
        t=t,
        language=language,
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
    port = int(os.environ.get('PORT', 5001))
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
