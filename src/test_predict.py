from predict import predict_image
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = ROOT / "data" / "Testing" / "glioma" / "Te-gl_10.jpg"

result = predict_image (TEST_FILE)

print("Predicted Class:", result["predicted_class"])
print("Prediction Score:", result["probability"])

print("\nAll scores:")

for class_name, score in result["all_scores"].items():
    print(f"{class_name}: {score:.4f}")