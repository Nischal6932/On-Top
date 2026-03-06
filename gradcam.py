import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("plant_disease_efficientnet.keras")

# Image path
image_path = "test_leaf.png"

# Load image
img = Image.open(image_path).convert("RGB").resize((224,224))
img_array = np.array(img)/255.0
img_array = np.expand_dims(img_array, axis=0)

# Get prediction
preds = model.predict(img_array)
class_index = np.argmax(preds[0])

# Find last convolution layer
last_conv_layer = None
for layer in reversed(model.layers):
    if "conv" in layer.name:
        last_conv_layer = layer.name
        break

# Create Grad-CAM model
grad_model = tf.keras.models.Model(
    [model.inputs],
    [model.get_layer(last_conv_layer).output, model.output]
)

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    loss = predictions[:, class_index]

grads = tape.gradient(loss, conv_outputs)

pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

conv_outputs = conv_outputs[0]

heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)

heatmap = np.maximum(heatmap,0) / np.max(heatmap)

# Resize heatmap
heatmap = cv2.resize(heatmap, (224,224))

# Convert heatmap to color
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# Overlay heatmap on image
original = cv2.imread(image_path)
original = cv2.resize(original,(224,224))

superimposed = heatmap * 0.4 + original

# Show result
plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(cv2.cvtColor(original,cv2.COLOR_BGR2RGB))

plt.subplot(1,3,2)
plt.title("Heatmap")
plt.imshow(heatmap)

plt.subplot(1,3,3)
plt.title("Grad-CAM")
plt.imshow(cv2.cvtColor(superimposed.astype("uint8"),cv2.COLOR_BGR2RGB))

plt.show()