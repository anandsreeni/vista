import cv2
import torch
from PIL import Image
from collections import Counter, deque
from transformers import CLIPModel, CLIPProcessor

# Edit/add to these freely — the model picks whichever description best
# matches the current camera frame, so more specific, distinct phrasing
# gives you more specific, distinct results.
PROMPTS = {
    "near staircase": "a photograph of stairs going up or down",
    "near hallway":   "a photograph of an empty long corridor",
    "near door":      "a photograph of a closed or open door up close",
    "inside a room":  "a photograph of the inside of a room with furniture",
    "near a wall":    "a photograph of a plain wall close up with no furniture",
    "near kitchen":   "a photograph of a kitchen with a counter, sink, or stove",
    "near windows":   "a photograph of windows letting in daylight",
    "classroom":      "a photograph of a classroom with desks and chairs",
}

class Localizer:
    def __init__(self, prompts=PROMPTS, min_conf=0.3, history=5):
        self.names = list(prompts.keys())
        self.min_conf = min_conf
        self.recent = deque(maxlen=history)

        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Precompute and store the tokenized prompts (reused every frame)
        text_inputs = self.processor(text=list(prompts.values()),
                                      return_tensors="pt", padding=True)
        self.input_ids = text_inputs["input_ids"]
        self.attention_mask = text_inputs["attention_mask"]

    def localize(self, frame):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_inputs = self.processor(images=img, return_tensors="pt")

        with torch.no_grad():
            out = self.model(
                input_ids=self.input_ids,
                attention_mask=self.attention_mask,
                pixel_values=img_inputs["pixel_values"],
                return_dict=True,
            )
            probs = out.logits_per_image.softmax(dim=-1)[0]

        print("ALL PROBS:", {n: round(float(p) * 100, 1) for n, p in zip(self.names, probs)})

        conf, idx = probs.max(dim=0)
        best = self.names[int(idx)]
        if float(conf) < self.min_conf:
            best = "unknown"

        self.recent.append(best)
        smoothed = Counter(self.recent).most_common(1)[0][0]
        return smoothed, int(float(conf) * 100)