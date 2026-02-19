
import vart
import xir
import numpy as np
import cv2
import time

class DPUInference:
    def __init__(self, xmodel_path):
        # Load compiled model
        self.graph = xir.Graph.deserialize(xmodel_path)
        self.subgraphs = self.graph.get_root_subgraph().children_topological_sort()
        
        # Get DPU subgraph
        self.dpu_subgraph = None
        for subgraph in self.subgraphs:
            if subgraph.get_attr("device") == "DPU":
                self.dpu_subgraph = subgraph
                break
        
        # Create DPU runner
        self.runner = vart.Runner.create_runner(
            self.dpu_subgraph, 
            "run"
        )
        
        # Get input/output tensors
        self.input_tensors = self.runner.get_input_tensors()
        self.output_tensors = self.runner.get_output_tensors()
        
        print(f"DPU model loaded successfully")
        print(f"Input tensors: {len(self.input_tensors)}")
        print(f"Output tensors: {len(self.output_tensors)}")
    
    def preprocess(self, image):
        """Preprocess for DPU"""
        # Get input tensor info
        input_tensor = self.input_tensors[0]
        input_shape = input_tensor.dims
        input_fixpos = input_tensor.get_attr("fix_point")
        
        # Resize
        h, w = input_shape[1], input_shape[2]
        img_resized = cv2.resize(image, (w, h))
        
        # Convert and normalize
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Quantize to int8 (DPU uses fixed-point)
        scale = 2 ** input_fixpos
        img_quantized = (img_rgb * scale).astype(np.int8)
        
        return img_quantized
    
    def inference(self, image):
        """Run DPU inference"""
        input_data = self.preprocess(image)
        
        # Prepare input/output buffers
        input_data = [input_data]
        output_data = [np.empty(tensor.dims, dtype=np.int8) 
                      for tensor in self.output_tensors]
        
        # Run inference
        start = time.perf_counter()
        job_id = self.runner.execute_async(input_data, output_data)
        self.runner.wait(job_id)
        end = time.perf_counter()
        
        latency_ms = (end - start) * 1000
        
        # Dequantize output
        output_tensor = self.output_tensors[0]
        output_fixpos = output_tensor.get_attr("fix_point")
        output_scale = 2 ** output_fixpos
        result = output_data[0].astype(np.float32) / output_scale
        
        return result, latency_ms
    
    def benchmark(self, test_images_dir, num_iterations=100):
        """Same benchmark structure as CPU baseline"""
        # [Similar to CPU benchmark code above]
        pass

if __name__ == "__main__":
    dpu = DPUInference("compiled_model/cnn_kv260.xmodel")
    dpu.benchmark("./test_images", 100)