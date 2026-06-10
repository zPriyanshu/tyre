import os
import json
import re
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Try to import MLX tools for optimized Mac support if requested and available
try:
    import mlx_vlm
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


class TyreOCREngine:
    def __init__(self, model_id="Qwen/Qwen2-VL-2B-Instruct", use_mlx=False):
        """
        Initializes the Tyre OCR Engine.
        
        Args:
            model_id (str): HuggingFace model identifier.
            use_mlx (bool): If True and on Apple Silicon, use MLX for faster inference.
        """
        self.model_id = model_id
        self.use_mlx = use_mlx and MLX_AVAILABLE
        self.model = None
        self.processor = None
        self.mlx_config = None
        self.device = None
        self.dtype = None

    def load_model(self):
        """
        Loads the model and processor into memory.
        """
        if self.use_mlx:
            print(f"Loading {self.model_id} via MLX for optimized Apple Silicon execution...")
            # Quantized MLX models are usually hosted under mlx-community
            # We map standard model ID to mlx-community counterpart if standard is passed
            mlx_model_id = self.model_id
            if "mlx-community" not in mlx_model_id:
                # E.g. Qwen/Qwen2-VL-2B-Instruct -> mlx-community/Qwen2-VL-2B-Instruct-4bit
                name = self.model_id.split("/")[-1]
                mlx_model_id = f"mlx-community/{name}-4bit"
            
            try:
                self.model, self.processor = mlx_vlm.load(mlx_model_id)
                self.mlx_config = mlx_vlm.utils.load_config(mlx_model_id)
                print("MLX Model loaded successfully.")
                return
            except Exception as e:
                print(f"Failed to load MLX model: {e}. Falling back to standard PyTorch/Transformers...")
                self.use_mlx = False

        # Determine best PyTorch device
        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.float16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            # Some M-series chips have better support for float16 over bfloat16
            self.dtype = torch.float16
        else:
            self.device = "cpu"
            self.dtype = torch.float32

        print(f"Loading {self.model_id} using PyTorch on device: {self.device} ({self.dtype})...")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        # Load model with optimal settings
        model_kwargs = {
            "torch_dtype": self.dtype,
        }
        
        if self.device != "cpu":
            # auto maps to available GPU/MPS automatically
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = None
            
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            **model_kwargs
        )
        
        if self.device == "cpu":
            # Explicitly move to CPU if no device map was set
            self.model = self.model.to(self.device)

        print("PyTorch Model loaded successfully.")

    def extract_tyre_info(self, image_path):
        """
        Extracts tyre specifications from an image.
        
        Args:
            image_path (str): Local path to the tyre image.
            
        Returns:
            dict: Parsed tyre details.
        """
        # Ensure model is loaded
        if self.model is None or self.processor is None:
            self.load_model()

        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")

        # Verify image format using PIL
        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception as e:
            raise ValueError(f"Invalid image format: {e}")

        # Define the structural extraction prompt
        prompt = (
            "You are an expert tyre inspector and OCR system.\n"
            "Analyze the provided image of a tyre sidewall and extract all visible markings and text.\n"
            "Focus on extracting the following tyre specifications:\n"
            "1. Brand/Manufacturer (e.g., Michelin, Bridgestone)\n"
            "2. Model/Pattern Name (e.g., Pilot Sport, Alenza, Turanza)\n"
            "3. Tyre Size (formatted as Width/Aspect Ratio R Rim Diameter, e.g., 205/55R16, 225/45R17, or 275/40ZR20)\n"
            "4. Load Index (numeric value representing load capacity, e.g., 91, 94, 101)\n"
            "5. Speed Rating (alphabetical character indicating maximum speed, e.g., H, V, W, Y)\n"
            "6. DOT Code (look for text starting with 'DOT' followed by 8 to 12 alphanumeric characters, especially the last 4 digits representing week and year of manufacture, e.g., 1221 for week 12 of 2021)\n"
            "7. Manufacturing Date (computed from the last 4 digits of the DOT code, format: MM/YYYY or 'Week XX, YYYY')\n"
            "8. Max Load and Pressure (e.g., 'MAX. LOAD 615 kg (1356 LBS)', 'MAX. PRESS. 340 kPa (50 PSI)')\n"
            "9. Other Markings (e.g., 'M+S', 'Tubeless', 'Radial', 'Extra Load / XL', 'Inside / Outside')\n"
            "10. All Sidewall Text (a list of all distinct text snippets visible on the tyre)\n\n"
            "Respond strictly in valid JSON format with the following keys:\n"
            "{\n"
            "  \"brand\": \"string or null\",\n"
            "  \"model\": \"string or null\",\n"
            "  \"tyre_size\": \"string or null\",\n"
            "  \"load_index\": \"string or null\",\n"
            "  \"speed_rating\": \"string or null\",\n"
            "  \"dot_code\": \"string or null\",\n"
            "  \"manufacturing_date\": \"string or null\",\n"
            "  \"max_load_pressure\": \"string or null\",\n"
            "  \"other_markings\": [\"string\"],\n"
            "  \"all_sidewall_text\": [\"string\"]\n"
            "}\n"
            "Do not include any introductory or concluding text. Return only the JSON."
        )

        if self.use_mlx:
            formatted_prompt = mlx_vlm.prompt_utils.apply_chat_template(
                self.processor, self.mlx_config, prompt, num_images=1
            )
            raw_output = mlx_vlm.generate(
                self.model, self.processor, formatted_prompt, [image_path], verbose=False
            )
        else:
            # Prepare inputs in HF chat format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Preprocess
            text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            
            # Move inputs to same device as model
            inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False  # Greedy decoding for consistent structured OCR
                )
                
            generated_ids = [ids[len(inputs["input_ids"][0]):] for ids in output_ids]
            raw_output = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]

        return self._clean_and_parse_json(raw_output)

    def _clean_and_parse_json(self, raw_output):
        """
        Parses raw model output into a valid Python dictionary, cleaning up markdown code blocks if present.
        """
        cleaned = raw_output.strip()
        
        # Remove markdown code fences if present
        if cleaned.startswith("```"):
            # Strip off the ```json or ``` at the start
            cleaned = re.sub(r"^```(?:json)?\n", "", cleaned)
            # Strip off the ``` at the end
            cleaned = re.sub(r"\n```$", "", cleaned)
            cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback parsing using regex if JSON is partially malformed
            print("Warning: Direct JSON parsing failed. Attempting cleanup extraction...")
            
            # Try to extract content between curly braces
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            
            # Create a simple fallback structure with raw text
            return {
                "brand": self._regex_extract(cleaned, r"brand[\"\']?\s*:\s*[\"\']([^\"\']+)"),
                "model": self._regex_extract(cleaned, r"model[\"\']?\s*:\s*[\"\']([^\"\']+)"),
                "tyre_size": self._regex_extract(cleaned, r"tyre_size[\"\']?\s*:\s*[\"\']([^\"\']+)"),
                "load_index": self._regex_extract(cleaned, r"load_index[\"\']?\s*:\s*[\"\']([^\"\']+)"),
                "speed_rating": self._regex_extract(cleaned, r"speed_rating[\"\']?\s*:\s*[\"\']([^\"\']+)"),
                "dot_code": self._regex_extract(cleaned, r"dot_code[\"\']?\s*:\s*[\"\']([^\"\']+)"),
                "manufacturing_date": self._regex_extract(cleaned, r"manufacturing_date[\"\']?\s*:\s*[\"\']([^\"\']+)"),
                "max_load_pressure": self._regex_extract(cleaned, r"max_load_pressure[\"\']?\s*:\s*[\"\']([^\"\']+)"),
                "other_markings": [],
                "all_sidewall_text": [raw_output]
            }

    def _regex_extract(self, text, pattern):
        """Helper to extract values from text if JSON parsing fails."""
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else None
