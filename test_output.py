import torch
from torchvision import transforms
from PIL import Image
import json

task = "../skincare"
idx = 1

# 1 find model.json, find input dimension and output dimension in ["inputs"][0]["tensor"]["dimensions"] and ["outputs"][0]["tensor"]["dimensions"]
# 2 load model weight from ["model_file"], remove prefix assets:///
with open(f"{task}/model.json", "r") as f:
    model_json = json.load(f)
    input_dim = model_json["inputs"][0]["tensor"]["dimensions"][::-1]
    output_dim = model_json["outputs"][0]["tensor"]["dimensions"][::-1]
    model_file = model_json["model_file"].replace("assets:///", "")
    model_path = f"{task}/{model_file}"
with open(f"{task}/labels.txt", "r") as f:
    class_names = f.read().splitlines()


# Load the TorchScript model
model = torch.jit.load(model_path)  # Replace 'model.pt' with your actual model file
model.eval()  # Set the model to evaluation mode

# Load and preprocess the image
image = Image.open(f"{task}_test_img_{idx}.jpeg").convert("RGB")  # Replace 'image.jpg' with your image file

# Define transformations: resizing, converting to tensor, and normalization (optional)
transform = transforms.Compose([
    transforms.Resize(input_dim[2:]),  # Resize to the model's required input size
    transforms.ToTensor(),          # Convert to a tensor and scales [0, 255] to [0.0, 1.0]
])
input_tensor = transform(image).unsqueeze(0)

# Pass the input through the model
with torch.no_grad():
    output = model(input_tensor)

# Print the raw output
print(f"Raw output: {output.numpy()}")
# print numpy with class names, sigmoid prob
output = torch.sigmoid(output)
output = output.squeeze(0).numpy()
for idx, prob in enumerate(output):
    print(f"{class_names[idx]}: {prob:.5f}")