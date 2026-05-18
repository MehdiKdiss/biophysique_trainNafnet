import matplotlib
matplotlib.use('Agg') # Prevent Tkinter conflict
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from PIL import Image, ImageTk, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import cv2
import threading
import torchvision.transforms as transforms
import torchvision.models as models
import anemia_segmentation  # Helper module for Cellpose/Anemia analysis

# ==========================================================
# 1) MODEL ARCHITECTURE (Must match trainingNafnet.py)
# ==========================================================
class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return x

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class SimplifiedChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Conv2d(channels, channels, 1, bias=True)
    
    def forward(self, x):
        y = F.adaptive_avg_pool2d(x, 1)
        y = self.fc(y)
        return x * y

class NAFBlock(nn.Module):
    def __init__(self, channels, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw_channels = channels * dw_expand
        
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, 1)
        self.conv2 = nn.Conv2d(dw_channels, dw_channels, 3, padding=1, groups=dw_channels)
        self.sg = SimpleGate()
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, 1)
        
        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_expand * channels, 1)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_expand * channels // 2, channels, 1)
        
        self.sca = SimplifiedChannelAttention(channels)
        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)))
    
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.conv3(x)
        x = self.sca(x)
        x = shortcut + x * self.beta
        
        shortcut = x
        x = self.norm2(x)
        x = self.conv4(x)
        x = self.sg2(x)
        x = self.conv5(x)
        x = shortcut + x * self.gamma
        
        return x

class NAFNet(nn.Module):
    def __init__(self, img_channels=3, width=48, middle_blocks=8, 
                 enc_blocks=[2, 2, 4, 8], dec_blocks=[2, 2, 2, 2]):
        super().__init__()
        
        self.intro = nn.Conv2d(img_channels, width, 3, padding=1)
        
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, 2, stride=2))
            chan *= 2
        
        self.middle = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blocks)])
        
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for num in dec_blocks:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1),
                nn.PixelShuffle(2)
            ))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            
        self.ending = nn.Conv2d(width, img_channels, 3, padding=1)
    
    def forward(self, x):
        x = self.intro(x)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        
        x = self.middle(x)
        
        for decoder, up, enc_skip in zip(self.decoders, self.ups, reversed(encs)):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
            
        x = self.ending(x)
        return x

# ==========================================================
# 2) GUI APPLICATION
# ==========================================================
class NAFNetApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window setup
        self.title("NAFNet Image Restoration Interface")
        self.geometry("1800x1000")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        # Device config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Model Loading
        self.load_model()

        # Variables
        self.original_image = None
        self.degraded_image = None
        self.restored_image = None
        self.img_path = None
        
        # Segmentation Model (Lazy load)
        self.seg_model = None
        self.seg_result_image = None
        self.seg_report = ""

        # GUI Layout
        self.create_widgets()

    def load_model(self):
        try:
            # Config matching trainingNafnet.py
            self.model = NAFNet(width=48, middle_blocks=8).to(self.device)
            
            model_path = "best_nafnet.pth"
            if os.path.exists(model_path):
                weights = torch.load(model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(weights)
                self.model.eval()
                print("Model loaded successfully!")
            else:
                tk.messagebox.showerror("Error", f"Model file '{model_path}' not found!")
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to load model: {e}")
            print(e)

    def create_widgets(self):
        # Configure grid layout (2 columns)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # === Sidebar (Controls) ===
        self.sidebar_frame = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        
        # Give the TextBox row (15) weight so it expands to fill space
        self.sidebar_frame.grid_rowconfigure(15, weight=1) 

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="NAFNet Control", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Basic Controls
        self.btn_load = ctk.CTkButton(self.sidebar_frame, text="Load Image", command=self.load_image)
        self.btn_load.grid(row=1, column=0, padx=20, pady=10)

        self.label_degrade = ctk.CTkLabel(self.sidebar_frame, text="Degradation Tools:", anchor="w")
        self.label_degrade.grid(row=2, column=0, padx=20, pady=(20, 0))

        # Noise Button
        self.btn_add_noise = ctk.CTkButton(self.sidebar_frame, text="Ajouter Bruit (Noise)", fg_color="gray", command=self.add_noise)
        self.btn_add_noise.grid(row=3, column=0, padx=20, pady=10)

        # Flow/Blur Button
        self.btn_add_flow = ctk.CTkButton(self.sidebar_frame, text="Ajouter Flow (Blur)", fg_color="gray", command=self.add_blur)
        self.btn_add_flow.grid(row=4, column=0, padx=20, pady=10)

        # Reset Dgradations
        self.btn_reset = ctk.CTkButton(self.sidebar_frame, text="Reset to Original", fg_color="darkred", command=self.reset_image)
        self.btn_reset.grid(row=5, column=0, padx=20, pady=10)

        # Restore
        self.label_restore = ctk.CTkLabel(self.sidebar_frame, text="Restoration:", anchor="w")
        self.label_restore.grid(row=6, column=0, padx=20, pady=(20, 0))

        self.btn_restore = ctk.CTkButton(self.sidebar_frame, text="TEST / RESTORE", fg_color="green", command=self.run_inference)
        self.btn_restore.grid(row=7, column=0, padx=20, pady=10)

        self.btn_save_restored = ctk.CTkButton(self.sidebar_frame, text="Save Restored Image", fg_color="#1F618D", command=self.save_restored)
        self.btn_save_restored.grid(row=8, column=0, padx=20, pady=10)

        # Exit Button
        self.btn_exit = ctk.CTkButton(self.sidebar_frame, text="EXIT APP", fg_color="black", hover_color="#330000", command=self.destroy)
        self.btn_exit.grid(row=9, column=0, padx=20, pady=(20, 10))

        # Metrics display
        self.psnr_label = ctk.CTkLabel(self.sidebar_frame, text="PSNR: N/A", font=ctk.CTkFont(size=16))
        self.psnr_label.grid(row=9, column=0, padx=20, pady=20)

        # Anemia Analysis Section
        self.label_anemia = ctk.CTkLabel(self.sidebar_frame, text="Anemia Analysis:", anchor="w")
        self.label_anemia.grid(row=10, column=0, padx=20, pady=(20, 0))

        self.btn_analyze = ctk.CTkButton(self.sidebar_frame, text="ANALYZE CELLS", fg_color="#D35400", command=self.start_anemia_analysis_thread)
        self.btn_analyze.grid(row=13, column=0, padx=20, pady=10)

        self.btn_save_seg = ctk.CTkButton(self.sidebar_frame, text="Save Analysis Result", fg_color="#A04000", command=self.save_segmentation)
        self.btn_save_seg.grid(row=14, column=0, padx=20, pady=10)

        self.txt_report = ctk.CTkTextbox(self.sidebar_frame, width=280, height=400)
        self.txt_report.grid(row=15, column=0, padx=10, pady=10, sticky="nsew")
        self.txt_report.insert("0.0", "Analysis Report will appear here...")
        self.txt_report.configure(state="disabled")

        # === Main Area (Images) ===
        self.main_frame = ctk.CTkScrollableFrame(self, label_text="Image Visualization")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        # 2x2 Grid
        self.main_frame.grid_columnconfigure((0, 1), weight=1)
        self.main_frame.grid_rowconfigure((0, 1, 2, 3), weight=0) # Expanding rows is tricky in scrollable

        # Image Panels Headers Row 1 (Inputs)
        self.panel_orig = ctk.CTkLabel(self.main_frame, text="1. Original (Reference)")
        self.panel_orig.grid(row=0, column=0, padx=10, pady=(10,0))
        
        self.panel_degraded = ctk.CTkLabel(self.main_frame, text="2. Degraded (Input)")
        self.panel_degraded.grid(row=0, column=1, padx=10, pady=(10,0))

        # Images Row 1
        self.img_label_orig = tk.Label(self.main_frame, text="[Empty]", width=50, height=20, bg="#2b2b2b", fg="white")
        self.img_label_orig.grid(row=1, column=0, padx=10, pady=10)

        self.img_label_degraded = tk.Label(self.main_frame, text="[Empty]", width=50, height=20, bg="#2b2b2b", fg="white")
        self.img_label_degraded.grid(row=1, column=1, padx=10, pady=10)

        # Image Panels Headers Row 2 (Outputs)
        self.panel_restored = ctk.CTkLabel(self.main_frame, text="3. Restored (Result)")
        self.panel_restored.grid(row=2, column=0, padx=10, pady=(20,0))

        self.panel_segmentation = ctk.CTkLabel(self.main_frame, text="4. Segmentation & Analysis")
        self.panel_segmentation.grid(row=2, column=1, padx=10, pady=(20,0))

        # Images Row 2
        self.img_label_restored = tk.Label(self.main_frame, text="[Empty]", width=50, height=20, bg="#2b2b2b", fg="white")
        self.img_label_restored.grid(row=3, column=0, padx=10, pady=10)

        self.img_label_segmentation = tk.Label(self.main_frame, text="[No Analysis]", width=50, height=20, bg="#2b2b2b", fg="white")
        self.img_label_segmentation.grid(row=3, column=1, padx=10, pady=10)


    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            self.img_path = file_path
            # Load and Resize for display logic would go here, 
            # but for processing we keep original size or resize strictly to model expectation
            pil_img = Image.open(file_path).convert("RGB")
            # For this specific model, training was on 256x256, but let's try to keep it flexible or resize
            # NAFNet is fully convolutional but local attention might expect divisible dimensions
            # We'll resize to 256x256 for consistent testing as per training script
            # We'll NOT resize to 512x512 anymore, to preserve original scale for segmentation analysis.
            # pil_img = pil_img.resize((512, 512), Image.BICUBIC)
            
            self.original_image = pil_img
            self.degraded_image = pil_img.copy()
            self.restored_image = None
            
            # Reset Segmentation Results
            self.seg_result_image = None
            self.seg_report = ""
            
            self.update_display()
            self.psnr_label.configure(text="PSNR: N/A")
            
            # Clear Report Textbox
            self.txt_report.configure(state="normal")
            self.txt_report.delete("0.0", "end")
            self.txt_report.insert("0.0", "Analysis Report will appear here...")
            self.txt_report.configure(state="disabled")

    def reset_image(self):
        if self.original_image:
            self.degraded_image = self.original_image.copy()
            self.restored_image = None
            
            # Reset Segmentation
            self.seg_result_image = None
            self.seg_report = ""
            
            self.update_display()
            self.psnr_label.configure(text="PSNR: N/A")
            
            # Clear Report
            self.txt_report.configure(state="normal")
            self.txt_report.delete("0.0", "end")
            self.txt_report.insert("0.0", "Analysis Report will appear here...")
            self.txt_report.configure(state="disabled")

    def add_noise(self):
        if self.degraded_image:
            # Add Gaussian Noise
            img_arr = np.array(self.degraded_image).astype(np.float32) / 255.0
            noise = np.random.normal(0, 0.05, img_arr.shape) # std=0.05 matches training
            img_noised = img_arr + noise
            img_noised = np.clip(img_noised, 0, 1)
            self.degraded_image = Image.fromarray((img_noised * 255).astype(np.uint8))
            self.update_display()

    def add_blur(self):
        if self.degraded_image:
            # Add Gaussian Blur (Simulating "Flow" / Flou)
            self.degraded_image = self.degraded_image.filter(ImageFilter.GaussianBlur(radius=2))
            self.update_display()


    def run_inference(self):
        print("Starting inference...")
        import gc
        gc.collect() # Helper for memory/tkinter handles
        
        try:
            if self.degraded_image is None:
                print("Degraded image is None.")
                tk.messagebox.showwarning("Warning", "Please load an image first.")
                return

            # Prepare input
            input_img = self.degraded_image
            w, h = input_img.size
            print(f"Input image size: {w}x{h}")
            
            # Padding to multiple of 32 (NAFNet downsamples 4 times -> 16, taking 32 for safety)
            # Or exactly size that matches encoder downs.
            # Downsampling factor = 2^len(enc_blocks) = 2^4 = 16.
            factor = 32
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            
            img_np = np.array(input_img)
            # Pad with reflection to avoid boundary artifacts
            img_padded = np.pad(img_np, ((0, padh), (0, padw), (0, 0)), 'reflect')
            
            img_tensor = torch.from_numpy(img_padded).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(self.device)

            # Inference
            print(f"Running model (Padded: {W}x{H})...")
            with torch.no_grad():
                output_tensor = self.model(img_tensor)
                output_tensor = torch.clamp(output_tensor, 0, 1)
            print("Model run complete.")

            # Convert back & Unpad
            output_arr = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            output_arr = output_arr[:h, :w, :] # Crop back to original
            
            self.restored_image = Image.fromarray((output_arr * 255).astype(np.uint8))
            print("Restored image created.")

            # Calculate PSNR
            try:
                from skimage.metrics import peak_signal_noise_ratio as compare_psnr
                # PSNR between Restored and Original
                orig_arr = np.array(self.original_image)
                rest_arr = np.array(self.restored_image)
                psnr_val = compare_psnr(orig_arr, rest_arr)
                self.after(0, lambda: self.psnr_label.configure(text=f"PSNR: {psnr_val:.2f} dB"))
            except ImportError:
                self.after(0, lambda: self.psnr_label.configure(text="PSNR: (skimage not found)"))
            except Exception as e:
                print(f"PSNR Error: {e}")
                
            # Schedule display update on main loop safely
            self.after(0, self.update_display)
            print("Display update scheduled.")
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            tk.messagebox.showerror("Inference Error", f"Failed: {e}")

    def update_display(self):
        # Fallback to standard ImageTk.PhotoImage if CTkImage is unstable
        def get_tkinter_image(pil_img):
            if pil_img is None:
                return None
            # Resize manually since we aren't using CTkImage's auto-scaling
            img_resized = pil_img.copy()
            # 450x450 - Requested larger size, swapped positions logic also applies
            img_resized.thumbnail((450, 450), Image.Resampling.LANCZOS)
            # Explicitly attach to self (Main Window) to handle multiple roots
            return ImageTk.PhotoImage(img_resized, master=self)

        try:
            # Original
            if self.original_image:
                img = get_tkinter_image(self.original_image)
                self.img_label_orig.configure(image=img, text="", width=450, height=450)
                self.img_label_orig.image = img  # Keep reference
            else:
                 self.img_label_orig.configure(image=None, text="[Empty]", width=20)
                 self.img_label_orig.image = None
            
            # Degraded
            if self.degraded_image:
                img = get_tkinter_image(self.degraded_image)
                self.img_label_degraded.configure(image=img, text="", width=450, height=450)
                self.img_label_degraded.image = img
            else:
                self.img_label_degraded.configure(image=None, text="[Empty]", width=20)
                self.img_label_degraded.image = None
                
            # Restored
            if self.restored_image:
                img = get_tkinter_image(self.restored_image)
                self.img_label_restored.configure(image=img, text="", width=450, height=450)
                self.img_label_restored.image = img
            else:
                self.img_label_restored.configure(image=None, text="[Empty]", width=20)
                self.img_label_restored.image = None
                
            # Segmentation
            if self.seg_result_image:
                img = get_tkinter_image(self.seg_result_image)
                self.img_label_segmentation.configure(image=img, text="", width=450, height=450)
                self.img_label_segmentation.image = img
            else:
                self.img_label_segmentation.configure(image=None, text="[No Analysis]", width=20)
                self.img_label_segmentation.image = None
                
        except Exception as e:
            print(f"Error updating display: {e}")
            import traceback
            traceback.print_exc()

    def start_anemia_analysis_thread(self):
        # Prevent double click / multiple threads
        if hasattr(self, '_analysis_running') and self._analysis_running:
            return
            
        target_img = self.restored_image if self.restored_image else self.original_image
        if target_img is None:
            tk.messagebox.showwarning("Warning", "Please load an image first.")
            return

        self.btn_analyze.configure(state="disabled", text="Running...")
        self.txt_report.configure(state="normal")
        self.txt_report.delete("0.0", "end")
        self.txt_report.insert("0.0", "Loading model and analyzing...\nThis may take 30-60s on CPU.\nPlease wait...")
        self.txt_report.configure(state="disabled")
        
        self._analysis_running = True
        
        # Start Thread
        thread = threading.Thread(target=self.run_anemia_analysis_logic, args=(target_img,), daemon=True)
        thread.start()

    def load_classifier_model(self):
        """Loads the binary classifier (ResNet18)"""
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Recreate model structure exactly as in training
            model = models.resnet18(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.5), 
                nn.Linear(num_ftrs, 2)
            )
            
            model_path = os.path.join(os.path.dirname(__file__), "best_classifier.pth")
            if not os.path.exists(model_path):
                print(f"[WARN] Classifier model not found at {model_path}")
                return None
            
            weights = torch.load(model_path, map_location=device)
            model.load_state_dict(weights)
            model.to(device)
            model.eval()
            print("[INFO] Classifier loaded successfully.")
            return model
        except Exception as e:
            print(f"[ERROR] Failed to load classifier: {e}")
            return None

    def run_anemia_analysis_logic(self, target_img):
        """
        Background Thread Logic:
        1. Run Global Classifier (Normal vs Anemia)
        2. IF Anemia -> Run Cellpose Segmentation & Report
        3. IF Healthy -> Report Healthy & Stop
        """
        print("[DEBUG] run_anemia_analysis_logic started")
        try:
            # ==========================
            # STEP 1: BINARY CLASSIFICATION
            # ==========================
            print("[DEBUG] Step 1: Loading Classifier...")
            self.after(0, lambda: self.update_report_text("Step 1: AI Diagnostics...\n"))
            
            classifier = self.load_classifier_model()
            prediction_label = "Unknown"
            is_anemic = True 
            
            if classifier:
                print("[DEBUG] Classifier loaded. Preprocessing image...")
                # Preprocess
                preprocess = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                
                # Check image mode
                print(f"[DEBUG] Image mode: {target_img.mode}")
                if target_img.mode != "RGB":
                    target_img = target_img.convert("RGB")

                input_tensor = preprocess(target_img).unsqueeze(0)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                input_tensor = input_tensor.to(device)
                
                print("[DEBUG] Running inference...")
                with torch.no_grad():
                    outputs = classifier(input_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)
                    idx = preds.item()
                    prob_val = probs[0][idx].item()
                    
                    print(f"[DEBUG] Prediction Index: {idx} (0=Healthy, 1=Anemic), Prob: {prob_val}")
                    
                    # 0: Healthy, 1: Anemic
                    if idx == 0:
                        prediction_label = f"HEALTHY (Normal)"
                        is_anemic = False
                    else:
                        prediction_label = f"ANEMIA DETECTED"
                        is_anemic = True
                        
                msg = f"\n>>> AI DIAGNOSIS: {prediction_label}\n    Confidence: {prob_val:.1%}\n"
                print(f"[DEBUG] UI Message: {msg}")
                self.after(0, lambda: self.update_report_text(msg))
            else:
                print("[WARN] Classifier not loaded.")
                self.after(0, lambda: self.update_report_text("\n[WARN] Classifier model missing. Proceeding to segmentation...\n"))

            
            # ==========================
            # STEP 2: CONDITIONAL FLOW
            # ==========================
            if not is_anemic:
                print("[DEBUG] Diagnosis is Healthy. Stopping.")
                self.after(0, lambda: self.update_report_text("\n>>> DIAGNOSIS: HEALTHY.\n    No further segmentation required."))
                self.after(0, lambda: self.finish_analysis_ui())
                return

            # ELSE: PROCEED TO CELLPOSE
            print("[DEBUG] Diagnosis is Anemia. Starting Segmentation...")
            self.after(0, lambda: self.update_report_text("\nStep 2: Starting Cellular Analysis (Cellpose)...\n"))

            if self.seg_model is None:
                print("Loading Cellpose model in background thread...")
                self.seg_model = anemia_segmentation.load_model()

            img_np = np.array(target_img)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            print("[DEBUG] Calling segment_and_analyze...")
            img_processed, classified_cells, df_rbc = anemia_segmentation.segment_and_analyze(self.seg_model, img_bgr)
            print("[DEBUG] Segmentation complete.")
            
            viz_img = anemia_segmentation.visualize_results(img_processed, classified_cells, save_path=None)
            diagnosis, mcv, frag_rate = anemia_segmentation.interpret_anemia(df_rbc)
            
            # Prepare results for UI
            seg_pil = Image.fromarray(cv2.cvtColor(viz_img, cv2.COLOR_BGR2RGB))
            report_text = (
                f"=== REPORT ===\n"
                f"Diagnosis: {diagnosis}\n\n"
                f"MCV: {mcv:.2f} fL (Norme: 80-100)\n"
                f"Schistocytes: {frag_rate:.2f}% (Seuil alerte: >1%)\n"
                f"RBC Count: {len(df_rbc)}\n"
            )
            
            # Schedule UI Update on Main Thread
            self.after(0, lambda: self.finish_analysis(seg_pil, report_text))
            
        except Exception as e:
            err_msg = str(e)
            self.after(0, lambda: self.fail_analysis(err_msg))

    def update_report_text(self, text):
        """Helper to append text to the report textbox from background thread"""
        self.txt_report.configure(state="normal")
        self.txt_report.insert("end", text)
        self.txt_report.configure(state="disabled")

    def finish_analysis_ui(self):
        """Helper to Reset UI state if analysis stops early (e.g. Healthy)"""
        self.btn_analyze.configure(state="normal", text="ANALYZE CELLS")
        self._analysis_running = False

    def finish_analysis(self, seg_pil, report_text):
        self.seg_result_image = seg_pil
        self.seg_report = report_text
        
        self.update_display()
        
        self.txt_report.configure(state="normal")
        self.txt_report.delete("0.0", "end")
        self.txt_report.insert("0.0", self.seg_report)
        self.txt_report.configure(state="disabled")
        
        self.btn_analyze.configure(state="normal", text="ANALYZE CELLS")
        self._analysis_running = False

    def fail_analysis(self, error_msg):
        tk.messagebox.showerror("Analysis Failure", f"Error: {error_msg}")
        self.btn_analyze.configure(state="normal", text="ANALYZE CELLS")
        self.txt_report.configure(state="normal")
        self.txt_report.insert("end", f"\n[FAILED]: {error_msg}")
        self.txt_report.configure(state="disabled")
        self._analysis_running = False

    def save_restored(self):
        if self.restored_image is None:
            tk.messagebox.showwarning("Warning", "No restored image to save.")
            return
            
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if file_path:
            self.restored_image.save(file_path)
            tk.messagebox.showinfo("Success", f"Restored image saved to {file_path}")

    def save_segmentation(self):
        if self.seg_result_image is None:
            tk.messagebox.showwarning("Warning", "No segmentation result to save.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if file_path:
            self.seg_result_image.save(file_path)
            # Also save report?
            report_path = file_path.replace(".png", "_report.txt").replace(".jpg", "_report.txt")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(self.seg_report)
            
            tk.messagebox.showinfo("Success", f"Segmentation image saved to {file_path}\nReport saved to {report_path}")


if __name__ == "__main__":
    app = NAFNetApp()
    app.mainloop()
