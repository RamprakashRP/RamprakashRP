import sys
import random
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw, ImageFont
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def get_points(img_array, target_num):
    pts = []
    h, w = img_array.shape
    for y in range(h):
        for x in range(w):
            if img_array[y, x]:
                pts.append((x, y))
    
    if len(pts) == 0:
        return [(150, 170)] * target_num
        
    if len(pts) >= target_num:
        return random.sample(pts, target_num)
    else:
        pts = pts * (target_num // len(pts) + 1)
        return pts[:target_num]

def generate_animated_svg(image_path, rp_logo_path, output_path):
    print("Loading and segmenting image...")
    # Load image
    orig_img = Image.open(image_path).convert('RGB')
    
    w, h = orig_img.size
    target_ratio = 300 / 340
    if w / h > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        no_bg = orig_img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_ratio)
        no_bg = orig_img.crop((0, 0, w, new_h))
        
    no_bg = no_bg.resize((300, 340), Image.Resampling.LANCZOS)
    
    # Geometric body mask (Head + Shoulders)
    mask = Image.new('L', (300, 340), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((90, 20, 210, 180), fill=255)  # Head
    draw.ellipse((40, 140, 260, 400), fill=255) # Shoulders
    mask = mask.filter(ImageFilter.GaussianBlur(15))
    
    black_bg = Image.new('RGB', (300, 340), (0, 0, 0))
    black_bg.paste(no_bg, mask=mask)
    
    img = ImageOps.autocontrast(black_bg, cutoff=1)
    img = img.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
    img = ImageEnhance.Contrast(img).enhance(1.5)
    
    gray = img.convert('L')
    gray = Image.composite(gray, Image.new('L', gray.size, 0), mask)
    dithered = np.array(gray.convert('1'))
    
    portrait_pts = []
    for y in range(340):
        for x in range(300):
            if dithered[y, x]:
                portrait_pts.append((x, y))
                
    print(f"Total portrait dots: {len(portrait_pts)}")
    num_travellers = min(1500, len(portrait_pts))
    random.shuffle(portrait_pts)
    face_pts = portrait_pts[:num_travellers]
    static_pts = portrait_pts[num_travellers:]
    
    # 2. Process RP Logo
    print("Processing RP Logo...")
    try:
        rp_img = Image.open(rp_logo_path).convert('RGBA')
        # Simple extraction: non-transparent pixels
        alpha_rp = rp_img.split()[-1]
        bbox_rp = alpha_rp.getbbox()
        if bbox_rp:
            rp_img = rp_img.crop(bbox_rp)
        rp_img = rp_img.resize((150, 150), Image.Resampling.LANCZOS)
        # Center it
        rp_canvas = Image.new('1', (300, 340), 0)
        rp_gray = Image.composite(rp_img.convert('L'), Image.new('L', rp_img.size, 0), rp_img.split()[-1])
        rp_dithered = rp_gray.convert('1')
        rp_canvas.paste(rp_dithered, (75, 95))
        rp_arr = np.array(rp_canvas)
        rp_base_pts = get_points(rp_arr, num_travellers)
    except Exception as e:
        print(f"Error loading RP logo: {e}. Generating fallback.")
        rp_canvas = Image.new('1', (300, 340), 0)
        draw = ImageDraw.Draw(rp_canvas)
        draw.text((75, 120), "RP", fill=1, font=ImageFont.load_default())
        rp_arr = np.array(rp_canvas)
        rp_base_pts = get_points(rp_arr, num_travellers)
        
    # Sort RP points by X coordinate to color R red and P white
    rp_base_pts.sort(key=lambda p: p[0])
    # The first half is R, second half is P
    
    # 3. Process Code Logo < >
    print("Processing Code Logo...")
    code_canvas = Image.new('1', (300, 340), 0)
    draw = ImageDraw.Draw(code_canvas)
    draw.line([(100, 130), (50, 170), (100, 210)], fill=1, width=15)
    draw.line([(200, 130), (250, 170), (200, 210)], fill=1, width=15)
    code_arr = np.array(code_canvas)
    code_pts = get_points(code_arr, num_travellers)
    
    # 4. Generate Random Start points
    random_pts = [(random.randint(0, 300), random.randint(0, 340)) for _ in range(num_travellers)]
    
    # 5. Optimal Transport
    print("Running optimal transport Phase 1 (Face -> RP)...")
    start_arr = np.array(face_pts)
    rp_arr_pts = np.array(rp_base_pts)
    dist1 = cdist(start_arr, rp_arr_pts)
    _, col1 = linear_sum_assignment(dist1)
    rp_ordered = rp_arr_pts[col1]
    
    print("Running optimal transport Phase 2 (RP -> Code)...")
    code_arr_pts = np.array(code_pts)
    dist2 = cdist(rp_ordered, code_arr_pts)
    _, col2 = linear_sum_assignment(dist2)
    code_ordered = code_arr_pts[col2]
    
    print("Building SVG...")
    kt = "0; 0.1; 0.3; 0.4; 0.6; 0.7; 0.9; 1"
    
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 340" width="100%" height="100%">']
    svg.append('<rect width="100%" height="100%" fill="#000000"/>')
    
    # Static layers (drifting and fading out when RP logo appears)
    num_groups = 50
    static_groups = [[] for _ in range(num_groups)]
    for i, pt in enumerate(static_pts):
        static_groups[i % num_groups].append(f"M{pt[0]},{pt[1]}h1")
        
    for i, g in enumerate(static_groups):
        path_str = " ".join(g)
        dx = random.randint(-15, 15)
        dy = random.randint(-15, 15)
        svg.append(f'<path d="{path_str}" stroke="#ff0000" stroke-width="1.5" shape-rendering="crispEdges">')
        # Opacity: 0; 1; 1; 0; 0; 0; 0; 0
        svg.append(f'  <animate attributeName="opacity" values="0; 1; 1; 0; 0; 0; 0; 0" keyTimes="{kt}" dur="14s" repeatCount="indefinite" />')
        # Transform: random -> 0 -> 0 -> drift -> drift -> drift -> drift -> random
        svg.append(f'  <animateTransform attributeName="transform" type="translate" values="{dx*5},{dy*5}; 0,0; 0,0; {dx},{dy}; {dx},{dy}; {dx},{dy}; {dx},{dy}; {dx*5},{dy*5}" keyTimes="{kt}" dur="14s" repeatCount="indefinite" />')
        svg.append('</path>')
        
    # Travellers
    # Calculate RP color assignment (left half red, right half white)
    # We sorted rp_base_pts earlier, but we need to know the index in the original rp_base_pts.
    # Actually, we ordered rp_ordered to match face_pts. Let's just check the x coordinate!
    # The centroid of RP logo is around x=150.
    
    for i in range(num_travellers):
        rx, ry = random_pts[i]
        fx, fy = face_pts[i]
        px, py = rp_ordered[i]
        cx, cy = code_ordered[i]
        
        rp_color = "#ff0000" if px < 150 else "#ffffff"
        
        svg.append(f'<rect width="2" height="2" fill="#ff0000">')
        svg.append(f'  <animate attributeName="x" values="{rx}; {fx}; {fx}; {px}; {px}; {cx}; {cx}; {rx}" keyTimes="{kt}" dur="14s" repeatCount="indefinite" />')
        svg.append(f'  <animate attributeName="y" values="{ry}; {fy}; {fy}; {py}; {py}; {cy}; {cy}; {ry}" keyTimes="{kt}" dur="14s" repeatCount="indefinite" />')
        svg.append(f'  <animate attributeName="fill" values="#ff0000; #ff0000; #ff0000; {rp_color}; {rp_color}; #ffffff; #ffffff; #ff0000" keyTimes="{kt}" dur="14s" repeatCount="indefinite" />')
        svg.append('</rect>')
        
    svg.append('</svg>')
    
    with open(output_path, 'w') as f:
        f.write("\\n".join(svg))
    print(f"Saved {output_path}")

generate_animated_svg('IMG_2290.jpeg', 'rp-logo.png', 'dithered-animated.svg')
