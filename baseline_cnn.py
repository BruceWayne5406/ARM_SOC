# baseline_cpu.py
import onnxruntime as ort
import numpy as np
import cv2
import time
from pathlib import Path

class CPUBaseline:
    def __init__(self, model_path):
        # Create ONNX Runtime session
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        
        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"Model loaded: {model_path}")
        print(f"Input shape: {self.input_shape}")
        print(f"Input name: {self.input_name}")
    
    def preprocess(self, image):
        """Preprocess image to model input format"""
        # Resize to model input size
        h, w = self.input_shape[2], self.input_shape[3]
        img_resized = cv2.resize(image, (w, h))
        
        # Convert to RGB and normalize
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        # NCHW format: (1, C, H, W)
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        img_batch = np.expand_dims(img_transposed, axis=0)
        
        return img_batch
    
    def inference(self, image):
        """Run inference on single image"""
        input_data = self.preprocess(image)
        
        start = time.perf_counter()
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_data}
        )
        end = time.perf_counter()
        
        latency_ms = (end - start) * 1000
        return outputs[0], latency_ms
    
    def benchmark(self, test_images_dir, num_iterations=100):
        """Comprehensive benchmark"""
        print(f"\n{'='*50}")
        print("CPU BASELINE BENCHMARK")
        print(f"{'='*50}\n")
        
        # Load test images
        image_paths = list(Path(test_images_dir).glob('*.jpg'))
        image_paths.extend(list(Path(test_images_dir).glob('*.png')))
        
        if not image_paths:
            print("No test images found!")
            return
        
        print(f"Found {len(image_paths)} test images")
        
        # Warm-up
        print("\nWarm-up runs...")
        test_img = cv2.imread(str(image_paths[0]))
        for _ in range(10):
            self.inference(test_img)
        
        # Actual benchmark
        print(f"\nRunning {num_iterations} iterations...")
        latencies = []
        
        for i in range(num_iterations):
            img_idx = i % len(image_paths)
            image = cv2.imread(str(image_paths[img_idx]))
            
            _, latency = self.inference(image)
            latencies.append(latency)
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{num_iterations} - "
                      f"Current: {latency:.2f}ms")
        
        # Statistics
        latencies = np.array(latencies)
        
        print(f"\n{'='*50}")
        print("RESULTS")
        print(f"{'='*50}")
        print(f"Average Latency:  {np.mean(latencies):.2f} ms")
        print(f"Std Deviation:    {np.std(latencies):.2f} ms")
        print(f"Min Latency:      {np.min(latencies):.2f} ms")
        print(f"Max Latency:      {np.max(latencies):.2f} ms")
        print(f"Median Latency:   {np.median(latencies):.2f} ms")
        print(f"Throughput:       {1000/np.mean(latencies):.2f} FPS")
        print(f"{'='*50}\n")
        
        # Save results
        np.savetxt('cpu_baseline_results.csv', 
                   latencies, 
                   delimiter=',',
                   header='latency_ms',
                   comments='')
        print("Results saved to cpu_baseline_results.csv")
        
        return {
            'avg_latency': np.mean(latencies),
            'std_dev': np.std(latencies),
            'throughput_fps': 1000/np.mean(latencies)
        }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to ONNX model')
    parser.add_argument('--images', required=True, help='Test images directory')
    parser.add_argument('--iterations', type=int, default=100)
    args = parser.parse_args()
    
    baseline = CPUBaseline(args.model)
    baseline.benchmark(args.images, args.iterations)