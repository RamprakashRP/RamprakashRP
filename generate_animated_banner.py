import sys
import random
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw, ImageFont
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def generate_animated_svg(image_path, output_path):
    print("Loading and processing image...")
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    target_ratio = 300 / 340
    if w / h > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_ratio)
        img = img.crop((0, 0, w, new_h))
        
    img = img.resize((300, 340), Image.Resampling.LANCZOS)
    
    mask = Image.new('L', (300, 340), 255)
    for y in range(340):
        for x in range(300):
            dx, dy = x - 150, y - 170
            d = (dx*dx/(150*150) + dy*dy/(170*170))**0.5
            if d > 0.8:
                mask.putpixel((x,y), int(max(0, 255 - (d-0.8)*500)))
                
    img = ImageOps.autocontrast(img, cutoff=1)
    img = img.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
    img = ImageEnhance.Contrast(img).enhance(1.3)
    
    gray = img.convert('L')
    gray = Image.composite(gray, Image.new('L', gray.size, 0), mask)
    dithered = np.array(gray.convert('1'))
    
    # Collect all bright pixels
    portrait_pts = []
    for y in range(340):
        for x in range(300):
            if dithered[y, x]:
                portrait_pts.append((x, y))
                
    print(f"Total portrait dots: {len(portrait_pts)}")
    
    num_travellers = min(900, len(portrait_pts))
    random.shuffle(portrait_pts)
    traveller_starts = portrait_pts[:num_travellers]
    static_pts = portrait_pts[num_travellers:]
    
    # Generate < > symbol points
    print("Generating symbol points...")
    symbol_img = Image.new('1', (300, 340), 0)
    draw = ImageDraw.Draw(symbol_img)
    try:
        font = ImageFont.truetype("arial.ttf", 180)
    except:
        font = ImageFont.load_default()
        
    # Draw roughly manually if font fails
    # Let's just draw lines for < >
    draw.line([(100, 100), (30, 170), (100, 240)], fill=1, width=10)
    draw.line([(200, 100), (270, 170), (200, 240)], fill=1, width=10)
    
    symbol_pts = []
    sym_arr = np.array(symbol_img)
    for y in range(340):
        for x in range(300):
            if sym_arr[y, x]:
                symbol_pts.append((x, y))
                
    # Sample exactly num_travellers points from symbol_pts
    if len(symbol_pts) >= num_travellers:
        traveller_ends = random.sample(symbol_pts, num_travellers)
    else:
        # duplicate if necessary
        traveller_ends = symbol_pts * (num_travellers // len(symbol_pts) + 1)
        traveller_ends = traveller_ends[:num_travellers]
        
    print("Running optimal transport...")
    start_arr = np.array(traveller_starts)
    end_arr = np.array(traveller_ends)
    dist_matrix = cdist(start_arr, end_arr)
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    
    traveller_ends_ordered = end_arr[col_ind]
    
    print("Building SVG...")
    # Animation timeline: 10s loop
    # 0s - 3s: Portrait visible
    # 3s - 4s: Morph to symbol
    # 4s - 6s: Symbol visible
    # 6s - 7s: Morph back
    # 7s - 10s: Portrait visible
    kt = "0; 0.3; 0.4; 0.6; 0.7; 1"
    
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 340" width="100%" height="100%">']
    svg.append('<rect width="100%" height="100%" fill="#000000"/>')
    
    # 1. Static layers (grouped for performance)
    num_groups = 50
    static_groups = [[] for _ in range(num_groups)]
    for i, pt in enumerate(static_pts):
        static_groups[i % num_groups].append(f"M{pt[0]},{pt[1]}h1")
        
    for i, g in enumerate(static_groups):
        path_str = " ".join(g)
        dx = random.randint(-20, 20)
        dy = random.randint(-20, 20)
        # Fade out and translate during morph
        svg.append(f'<path d="{path_str}" stroke="#ff0000" stroke-width="1.5" shape-rendering="crispEdges">')
        svg.append(f'  <animate attributeName="opacity" values="1; 1; 0; 0; 1; 1" keyTimes="{kt}" dur="10s" repeatCount="indefinite" />')
        svg.append(f'  <animateTransform attributeName="transform" type="translate" values="0,0; 0,0; {dx},{dy}; {dx},{dy}; 0,0; 0,0" keyTimes="{kt}" dur="10s" repeatCount="indefinite" />')
        svg.append('</path>')
        
    # 2. Travellers
    for i in range(num_travellers):
        sx, sy = traveller_starts[i]
        ex, ey = traveller_ends_ordered[i]
        svg.append(f'<rect width="1.5" height="1.5" fill="#ffffff">')
        svg.append(f'  <animate attributeName="x" values="{sx}; {sx}; {ex}; {ex}; {sx}; {sx}" keyTimes="{kt}" dur="10s" repeatCount="indefinite" />')
        svg.append(f'  <animate attributeName="y" values="{sy}; {sy}; {ey}; {ey}; {sy}; {sy}" keyTimes="{kt}" dur="10s" repeatCount="indefinite" />')
        # Travelers start red (matching portrait) and turn white (for the code symbol)
        svg.append(f'  <animate attributeName="fill" values="#ff0000; #ff0000; #ffffff; #ffffff; #ff0000; #ff0000" keyTimes="{kt}" dur="10s" repeatCount="indefinite" />')
        svg.append('</rect>')
        
    svg.append('</svg>')
    
    with open(output_path, 'w') as f:
        f.write("\\n".join(svg))
    print(f"Saved {output_path}")

generate_animated_svg('IMG_2290.jpeg', 'dithered-animated.svg')
