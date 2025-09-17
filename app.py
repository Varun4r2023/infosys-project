import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr

# ----------------------------
# 1. Load the trained model
# ----------------------------
def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))

    # Extract class names and num_classes
    class_names = checkpoint.get("class_names", ["class0", "class1"])
    num_classes = checkpoint.get("num_classes", len(class_names))

    # Build resnet18 with correct num_classes
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Fix key mismatch (remove "model." prefix)
    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k.replace("model.", "", 1)] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    return model, class_names



# ⚠️ Add your model path here
MODEL_PATH = "C:/Users/admin/Downloads/final_resnet18_model.pth"


# Load model and class names
model, CLASS_NAMES = load_model(MODEL_PATH)


# ----------------------------
# 2. Image preprocessing
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ----------------------------
# 3. Prediction function
# ----------------------------
def predict(image):
    image = Image.fromarray(image).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        top3_prob, top3_idx = torch.topk(probs, 3)

    results = {CLASS_NAMES[i]: float(top3_prob[j]) for j, i in enumerate(top3_idx)}
    return results


# ----------------------------
# 4. Gradio Interface
# ----------------------------
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload Leaf Image"),
    outputs=gr.Label(num_top_classes=3, label="Top Predictions"),
    title="🌱 Plant Disease Detection",
    description="Upload an image of a plant leaf to detect possible diseases using a trained ResNet18 model."
)

# Launch app
if __name__ == "__main__":
    interface.launch()
