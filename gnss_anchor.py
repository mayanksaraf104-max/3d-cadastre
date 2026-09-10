import numpy as np
from shapely.geometry import Polygon

class GNSSCoordinateAnchor:
    def __init__(self, gcp_pixels, gcp_real_world):
        """
        Initializes the GNSS Anchor by calculating the Affine Transformation Matrix.
        Requires at least 3 Ground Control Points (GCPs) to calculate Scaling, Rotation, and Translation.
        """
        self.matrix = self._calculate_affine_matrix(gcp_pixels, gcp_real_world)

    def _calculate_affine_matrix(self, pixels, real_world):
        """
        Uses Ordinary Least Squares (OLS) to solve the Affine Transformation:
        X_real = a*x_pixel + b*y_pixel + c
        Y_real = d*x_pixel + e*y_pixel + f
        """
        print("🛰️ Calculating GNSS/CORS Affine Transformation Matrix...")
        
        # Build the A matrix from pixel coordinates [x, y, 1]
        A = np.c_[pixels, np.ones(pixels.shape[0])]
        
        # Real-world X and Y vectors
        X_real = real_world[:, 0]
        Y_real = real_world[:, 1]
        
        # Solve for coefficients [a, b, c] and [d, e, f]
        coef_X, _, _, _ = np.linalg.lstsq(A, X_real, rcond=None)
        coef_Y, _, _, _ = np.linalg.lstsq(A, Y_real, rcond=None)
        
        return {
            'a': coef_X[0], 'b': coef_X[1], 'c': coef_X[2],
            'd': coef_Y[0], 'e': coef_Y[1], 'f': coef_Y[2]
        }

    def anchor_polygon(self, pixel_polygon):
        """
        Takes a Shapely polygon in pixel space and mathematically warps it 
        to real-world Geographic Coordinates (UTM meters).
        """
        real_world_coords = []
        for x, y in pixel_polygon.exterior.coords:
            # Apply the transformation matrix to every single vertex
            X = self.matrix['a'] * x + self.matrix['b'] * y + self.matrix['c']
            Y = self.matrix['d'] * x + self.matrix['e'] * y + self.matrix['f']
            real_world_coords.append((X, Y))
            
        return Polygon(real_world_coords)

# --- Hackathon Proof Test ---
if __name__ == "__main__":
    print("\n--- GNSS / CORS Anchoring Test ---")
    
    # 1. Simulated Pixel Coordinates from your Blueprint (e.g., corners of the image)
    pixels = np.array([
        [0, 0],         # Top Left
        [1000, 0],      # Top Right
        [0, 1000]       # Bottom Left
    ])
    
    # 2. Simulated Real-World CORS GPS Data (UTM Zone 10N - San Francisco)
    # UTM uses meters, which is perfect for OpenCASCADE True 3D geometry!
    utm_coords = np.array([
        [552000.0, 4182000.0],  # Real-world Top Left
        [552500.0, 4182000.0],  # Real-world Top Right (500m wide)
        [552000.0, 4181500.0]   # Real-world Bottom Left (500m tall)
    ])
    
    # Initialize the anchor
    gnss_rover = GNSSCoordinateAnchor(pixels, utm_coords)
    
    # Test it on an AI-extracted room
    dummy_ai_room = Polygon([(100, 100), (300, 100), (300, 300), (100, 300), (100, 100)])
    print(f"\n📏 Local Pixel Room Coordinates:\n   {list(dummy_ai_room.exterior.coords)[:2]}...")
    
    anchored_room = gnss_rover.anchor_polygon(dummy_ai_room)
    print(f"\n🌍 Anchored Global UTM Coordinates:\n   {list(anchored_room.exterior.coords)[:2]}...")