import os
import laspy
import numpy as np
from pathlib import Path

def create_mock_laz():
    output_path = Path("data/raw_lidar/san_francisco_3dep_sample.laz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("🛠️ Generating Ultra-Irregular True 3D LiDAR point cloud (Slant + Dome + Waves)...")
    
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scale = [0.001, 0.001, 0.001]
    header.offset = [0, 0, 0]
    
    las = laspy.LasData(header)
    
    num_points = 200000 
    
    # 1. Generate purely in NumPy first to avoid Laspy ScaledArrayView errors
    x_vals = np.random.uniform(0, 1000, num_points)
    y_vals = np.random.uniform(0, 1000, num_points)
    
    # =====================================================================
    # 🌪️ THE ULTIMATE IRREGULAR TOPOGRAPHY ENGINE
    # =====================================================================
    
    # A. Linear Base Slant
    slant = (0.01 * x_vals) + (0.005 * y_vals)
    
    # B. Polynomial Circular Dome
    dome = np.where(
        ((x_vals - 500)**2 + (y_vals - 500)**2) < 90000, 
        -0.0002 * (x_vals - 500)**2 - 0.0002 * (y_vals - 500)**2 + 18.0, 
        0
    )
    
    # C. Irregular Sinusoidal Waves
    waves = 2.5 * np.sin(x_vals / 40.0) + 2.5 * np.cos(y_vals / 40.0)
    
    # D. Simulating real-world laser scatter (noise)
    noise = np.random.normal(0, 0.15, num_points) 
    
    # Combine all mathematical shapes into one continuous Z-axis array
    z_vals = np.maximum(slant + dome + waves + noise, 3.0)
    
    # 2. Assign the calculated arrays back to the Laspy object at the very end
    las.x = x_vals
    las.y = y_vals
    las.z = z_vals
    
    las.write(output_path)
    print(f"✅ Successfully created Irregular True 3D LiDAR file at: {output_path}")

if __name__ == "__main__":
    create_mock_laz()