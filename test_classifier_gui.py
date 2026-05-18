import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import os

# Set appearance mode
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class ClassifierTestApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Anemia Classifier Tester")
        self.geometry("600x700")

        # Model placeholder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.load_model()

        # UI Layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0) # Header
        self.grid_rowconfigure(1, weight=1) # Image
        self.grid_rowconfigure(2, weight=0) # Controls
        self.grid_rowconfigure(3, weight=0) # Result

        # Header
        self.label_header = ctk.CTkLabel(self, text="Binary Classifier Test (Anemia vs Normal)", font=ctk.CTkFont(size=20, weight="bold"))
        self.label_header.grid(row=0, column=0, padx=20, pady=20)

        # Image Display
        self.img_label = ctk.CTkLabel(self, text="[No Image Loaded]", width=400, height=400, fg_color="gray20", corner_radius=10)
        self.img_label.grid(row=1, column=0, padx=20, pady=10)

        # Controls Frame
        self.frame_controls = ctk.CTkFrame(self)
        self.frame_controls.grid(row=2, column=0, padx=20, pady=20)

        self.btn_load = ctk.CTkButton(self.frame_controls, text="Load Image", command=self.load_image, font=ctk.CTkFont(size=16))
        self.btn_load.pack(side="left", padx=20)

        self.btn_predict = ctk.CTkButton(self.frame_controls, text="Run Prediction", command=self.run_prediction, fg_color="green", font=ctk.CTkFont(size=16))
        self.btn_predict.pack(side="left", padx=20)

        # Result Display
        self.label_result = ctk.CTkLabel(self, text="Result: Waiting...", font=ctk.CTkFont(size=24, weight="bold"), text_color="gray")
        self.label_result.grid(row=3, column=0, padx=20, pady=20)
        
        self.current_image = None
        self.current_filepath = None

    def load_model(self):
        try:
            print(f"Loading model on {self.device}...")
            # Recreate model structure
            self.model = models.resnet18(weights=None)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(0.5), 
                nn.Linear(num_ftrs, 2)
            )
            
            model_path = "best_classifier.pth"
            if not os.path.exists(model_path):
                tk.messagebox.showerror("Error", f"Model file '{model_path}' not found!")
                return

            weights = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(weights)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")
            
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to load model: {e}")

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tif")])
        if file_path:
            self.current_filepath = file_path
            
            # Load and display
            pil_img = Image.open(file_path).convert("RGB")
            self.current_image = pil_img
            
            # Resize for display (preserving aspect ratio approximately or just fit box)
            # Simple Thumbnail
            display_img = pil_img.copy()
            display_img.thumbnail((400, 400), Image.Resampling.LANCZOS)
            
            # Use Standard Window PhotoImage to avoid CTk bug if present
            tk_img = ImageTk.PhotoImage(display_img)
            self.img_label.configure(image=tk_img, text="")
            self.img_label.image = tk_img # Keep reference
            
            self.label_result.configure(text="Ready to Predict", text_color="white")

    def run_prediction(self):
        if self.model is None:
            tk.messagebox.showwarning("Warning", "Model not loaded.")
            return
        if self.current_image is None:
            tk.messagebox.showwarning("Warning", "Load an image first.")
            return

        try:
            # Preprocessing (Same as Validation)
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            input_tensor = preprocess(self.current_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                idx = preds.item()
                prob_val = probs[0][idx].item()

                if idx == 0:
                    result_text = f"HEALTHY (Normal)\nConfidence: {prob_val:.1%}"
                    color = "green"
                else:
                    result_text = f"ANEMIA DETECTED\nConfidence: {prob_val:.1%}"
                    color = "red"
                
                self.label_result.configure(text=result_text, text_color=color)
                print(f"Prediction: {result_text}")

        except Exception as e:
            tk.messagebox.showerror("Prediction Error", str(e))

if __name__ == "__main__":
    app = ClassifierTestApp()
    app.mainloop()
