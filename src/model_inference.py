import os
from diffusers import StableDiffusionPipeline
import torch
import GPUtil
import time

class GenAI:
    
    def __init__(self, model_id="CompVis/stable-diffusion-v1-4", model_path=None, log_progress=False):
        
        # # Device Selection
        # if torch.cuda.is_available():
        #     self.device = "cuda"
        # else:
        #     self.device = "cpu"

        # # Default Model Parameters
        # self.default_seed = 42

        # # VRAM Allocation (For GPUs)
        # if self.device == "cuda":
        #     self.vram = self.get_available_vram()
        # else:
        #     self.vram = 0
        
        # # SD Pipeline
        # self.pipe = self.create_pipeline()

        self.__log_progress = log_progress
        self.__default_seed = 42
        self.__model_id = model_id
        self.__model_path = model_path

        self.__device = self.__get_compute_device()
        self.__vram = self.__get_available_vram() if self.__device in ["cuda", "hip"] else 0
        self.__pipe = self.__create_pipeline()


    # def get_available_vram(self):
    #     gpu_list = GPUtil.getGPUs()
    #     if gpu_list:
    #         return gpu_list[0].memoryTotal / 1024
    #     return 0


    # def create_pipeline(self):

    #     # Set Torch Data Type
    #     if self.device == "cuda":
    #         dtype = torch.float16
    #     else:
    #         dtype = torch.float32

    #     # Hugging Face Model ID
    #     model_id = self.model

    #     # Pre-Trained txt2Img Pipeline
    #     pipe = StableDiffusionPipeline.from_pretrained(
    #         model_id,
    #         torch_dtype = dtype
    #     ).to(self.device)

    #     # Optimization
    #     pipe.enable_attention_slicing()

    #     return pipe
    

    # def get_model_parameters(self):
    #     if self.device == "cuda":
    #         if self.vram >= 20:
    #             return 1024, 1024, 50, 8.0
    #         elif self.vram >= 12:
    #             return 768, 768, 40, 7.5
    #         elif self.vram >= 6:
    #             return 640, 640, 35, 7.0
    #         elif self.vram >= 4:
    #             return 512, 512, 30, 6.5
    #         else:
    #             return 384, 384, 25, 6.0
    #     else:
    #         return 384, 384, 25, 6.0
        
    
    # def generate_image(self, promptText, outputPath):
        
    #     try:

    #         # Print Device Info
    #         print(f"Compute Device : {self.device.upper()}")

    #         if self.device == "cuda":
    #             print(f"VRAM           : {self.vram:.2f} GB")

    #         # Start Timer
    #         start_time = time.time()

    #         # Get Model Parameters
    #         width, height, steps, scale = self.get_model_parameters()

    #         # Define Torch Generator Object
    #         generator = torch.Generator(self.device).manual_seed(self.default_seed)

    #         # Generate Image
    #         image = self.pipe(
    #             promptText,
    #             height = height,
    #             width = width,
    #             num_inference_steps = steps,
    #             guidance_scale = scale,
    #             generator = generator
    #         ).images[0]

    #         # Stop Timer
    #         end_time = time.time()
    #         elapsed_time = end_time - start_time

    #         # Save Image To Specified Output Path From Client
    #         image.save(outputPath, format="PNG")
            
            
    #         print(f"Time Required  : {elapsed_time:.2f} seconds")

    #     except Exception as e:
    #         return f"Exception in GenerateImage(): {str(e)}"

    def __log(self, message):
        if self.__log_progress:
            print(f"[GenAI] {message}")

    def __get_compute_device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.version, "hip") and torch.version.hip is not None:
            return "hip"
        else:
            return "cpu"

    def __get_available_vram(self):
        try:
            gpu_list = GPUtil.getGPUs()
            if gpu_list:
                return gpu_list[0].memoryFree / 1024
        except:
            self.__log("GPUtil failed. Estimating VRAM may not work on ROCm.")
        return 0

    def __create_pipeline(self):
        dtype = torch.float16 if self.__device == "cuda" else torch.float32
        self.__log(f"Loading model: {self.__model_id} on {self.__device.upper()}")

        kwargs = {
            "torch_dtype": dtype,
        }
        if self.__model_path:
            kwargs["cache_dir"] = self.__model_path  # Local download path

        pipe = StableDiffusionPipeline.from_pretrained(
            self.__model_id,
            **kwargs
        ).to(self.__device)

        if self.__device in ["cuda", "hip"]:
            pipe.enable_attention_slicing()

        return pipe

    def __get_model_parameters(self):
        if self.__device in ["cuda", "hip"]:
            if self.__vram >= 20:
                return 1024, 1024, 50, 8.0
            elif self.__vram >= 12:
                return 768, 768, 40, 7.5
            elif self.__vram >= 6:
                return 640, 640, 35, 7.0
            elif self.__vram >= 4:
                return 512, 512, 30, 6.5
            else:
                self.__log("Low VRAM: Falling back to minimal settings.")
                return 384, 384, 25, 6.0
        else:
            return 384, 384, 25, 6.0

    def generate_image(self, promptText, outputPath, seed=None, num_images=1, image_format="PNG"):
        try:
            start_time = time.time()
            width, height, steps, scale = self.__get_model_parameters()
            seed = seed or self.__default_seed
            generator = torch.Generator(self.__device).manual_seed(seed)

            self.__log(f"Generating {num_images} Image(s)...")

            results = self.__pipe(
                [promptText] * num_images,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=scale,
                generator=generator
            ).images

            saved_files = []
            for idx, img in enumerate(results):
                filename = outputPath if num_images == 1 else f"{os.path.splitext(outputPath)[0]}_{idx+1}.{image_format.lower()}"
                img.save(filename, format=image_format.upper())
                saved_files.append(filename)

            elapsed_time = time.time() - start_time

            report_lines = [
                f"Compute Device : {self.__device.upper()}",
                f"VRAM           : {self.__vram:.2f} GB" if self.__device in ["cuda", "hip"] else "",
                f"Images Created : {num_images}",
                f"Time Required  : {elapsed_time:.2f} seconds",
                f"Saved To       : {', '.join(saved_files)}"
            ]

            self.__log("\n".join([line for line in report_lines if line]))

        except Exception as e:
            return f"Exception in generate_image(): {str(e)}"
