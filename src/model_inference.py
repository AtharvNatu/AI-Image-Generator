from diffusers import StableDiffusionPipeline
import torch
import GPUtil
import time

class StableDiffusionServer:
    def __init__(self):
        
        # Device Selection
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Default Model Parameters
        self.default_seed = 42

        # VRAM Allocation (For GPUs)
        if self.device == "cuda":
            self.vram = self.get_available_vram()
        else:
            self.vram = 0
        
        # SD Pipeline
        self.pipe = self.create_pipeline()


    def get_available_vram(self):
        gpu_list = GPUtil.getGPUs()
        if gpu_list:
            return gpu_list[0].memoryTotal / 1024
        return 0


    def create_pipeline(self):

        # Set Torch Data Type
        if self.device == "cuda":
            dtype = torch.float16
        else:
            dtype = torch.float32

        # Hugging Face Model ID
        model_id = "CompVis/stable-diffusion-v1-4"

        # Pre-Trained txt2Img Pipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype = dtype
        ).to(self.device)

        # Optimization
        pipe.enable_attention_slicing()

        return pipe
    

    def get_model_parameters(self):
        if self.device == "cuda":
            if self.vram >= 20:
                return 1024, 1024, 50, 8.0
            elif self.vram >= 12:
                return 768, 768, 40, 7.5
            elif self.vram >= 6:
                return 640, 640, 35, 7.0
            elif self.vram >= 4:
                return 512, 512, 30, 6.5
            else:
                return 384, 384, 25, 6.0
        else:
            return 384, 384, 25, 6.0
        
    
    def GenerateImage(self, promptText, outputPath):
        
        try:

            # Start Timer
            start_time = time.time()

            # Get Model Parameters
            width, height, steps, scale = self.get_model_parameters()

            # Define Torch Generator Object
            generator = torch.Generator(self.device).manual_seed(self.default_seed)

            # Generate Image
            image = self.pipe(
                promptText,
                height = height,
                width = width,
                num_inference_steps = steps,
                guidance_scale = scale,
                generator = generator
            ).images[0]

            # Stop Timer
            end_time = time.time()
            elapsed_time = end_time - start_time

            # Save Image To Specified Output Path From Client
            image.save(outputPath, format="PNG")
            
            if self.device == "cuda":
                return "\n".join([
                    f"Compute Device : {self.device.upper()}",
                    f"VRAM           : {self.vram:.2f} GB",
                    f"Time Required  : {elapsed_time:.2f} seconds"
                ])
            else:
                return "\n".join([
                    f"Compute Device : {self.device.upper()}",
                    f"Time Required  : {elapsed_time:.2f} seconds"
                ])

        except Exception as e:
            return f"Exception in GenerateImage(): {str(e)}"

