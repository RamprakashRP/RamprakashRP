import sys
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np

def generate_svg(image_path, output_path):
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # 1. Crop to head and shoulders (assume center-top crop)
    w, h = img.size
    target_ratio = 300 / 340
    if w / h > target_ratio:
        # Image is wider
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    else:
        # Image is taller
        new_h = int(w / target_ratio)
        top = 0 # keep head
        img = img.crop((0, top, w, top + new_h))
        
    img = img.resize((300, 340), Image.Resampling.LANCZOS)
    
    # 2. Segment background (simple thresholding based on corners)
    # The user's image might have a complex background, but we'll approximate 
    # by making the edges darker using a vignette mask to blend into the black background
    mask = Image.new('L', (300, 340), 255)
    for y in range(340):
        for x in range(300):
            # Distance from center-ish (150, 150)
            dx = x - 150
            dy = y - 170
            d = (dx*dx/(150*150) + dy*dy/(170*170))**0.5
            if d > 0.8:
                mask.putpixel((x,y), int(max(0, 255 - (d-0.8)*500)))
                
    # 3. Contrast, Unsharp, Autocontrast
    img = ImageOps.autocontrast(img, cutoff=1)
    img = img.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.3)
    
    # 4. Convert to grayscale and apply mask
    gray = img.convert('L')
    gray = Image.composite(gray, Image.new('L', gray.size, 0), mask)
    
    # 5. Floyd-Steinberg Dither
    dithered = gray.convert('1')
    
    # 6. Generate SVG path
    # We want Red dots on the BRIGHT areas to avoid the negative look
    # In a '1' mode image, 255 is white (bright), 0 is black (dark)
    pixels = np.array(dithered)
    
    path_data = []
    # Serpentine iteration
    for y in range(340):
        row = range(300) if y % 2 == 0 else range(299, -1, -1)
        start_x = None
        for x in row:
            if pixels[y, x] == 255: # White/Bright pixel -> Draw Red Dot
                if start_x is None:
                    start_x = x
            else:
                if start_x is not None:
                    # End of a run
                    if start_x <= x:
                        path_data.append(f"M{start_x},{y}h{x - start_x}")
                    else:
                        path_data.append(f"M{x+1},{y}h{start_x - x}")
                    start_x = None
        if start_x is not None:
            if y % 2 == 0:
                path_data.append(f"M{start_x},{y}h{300 - start_x}")
            else:
                path_data.append(f"M{0},{y}h{start_x + 1}")
                
    path_str = " ".join(path_data)
    
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 340" width="100%" height="100%">
    <rect width="100%" height="100%" fill="#0d1117"/>
    <path d="{path_str}" stroke="#ff0000" stroke-width="1" shape-rendering="crispEdges"/>
</svg>'''
    
    with open(output_path, 'w') as f:
        f.write(svg)

generate_svg('IMG_2290.jpeg', 'dithered.svg')
