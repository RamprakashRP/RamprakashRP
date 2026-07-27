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
    # Load the PNG with transparent background
    orig_img = Image.open(image_path).convert('RGBA')
    
    # Crop based on alpha bounding box
    alpha = orig_img.split()[-1]
    bbox = alpha.getbbox()
    if bbox:
        orig_img = orig_img.crop(bbox)
        
    w, h = orig_img.size
    target_ratio = 300 / 340
    if w / h > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        img_cropped = orig_img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_ratio)
        img_cropped = orig_img.crop((0, 0, w, new_h))
        
    img_resized = img_cropped.resize((300, 340), Image.Resampling.LANCZOS)
    
    # We use the alpha channel directly as the mask!
    final_mask = img_resized.split()[-1]
    # Invert the alpha to be 0 for foreground and 255 for background (as used below)
    final_mask = ImageOps.invert(final_mask)
    
    black_bg = Image.new('RGB', (300, 340), (0, 0, 0))
    black_bg.paste(img_resized.convert('RGB'), mask=img_resized.split()[-1])
    
    img = ImageOps.autocontrast(black_bg, cutoff=1)
    img = img.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
    img = ImageEnhance.Contrast(img).enhance(1.5)
    
    gray = img.convert('L')
    gray = Image.composite(gray, Image.new('L', gray.size, 0), final_mask)
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
    
    print("Processing RP Logo...")
    rp_img = Image.open(rp_logo_path).convert('RGBA')
    rp_w, rp_h = rp_img.size
    scale = 150 / max(rp_w, rp_h)
    new_rp_w, new_rp_h = int(rp_w * scale), int(rp_h * scale)
    rp_img = rp_img.resize((new_rp_w, new_rp_h), Image.Resampling.LANCZOS)
    
    rp_canvas = Image.new('RGBA', (300, 340), (0, 0, 0, 0))
    offset_x = (300 - new_rp_w) // 2
    offset_y = (340 - new_rp_h) // 2
    rp_canvas.paste(rp_img, (offset_x, offset_y))
    
    rp_valid_pts = []
    rp_colors = {}
    for y in range(340):
        for x in range(300):
            r, g, b, a = rp_canvas.getpixel((x,y))
            if a > 128:
                rp_valid_pts.append((x, y))
                if r > 150 and g < 100 and b < 100:
                    rp_colors[(x,y)] = "#ff0000"
                elif r > 150 and g > 150 and b > 150:
                    rp_colors[(x,y)] = "#ffffff"
                else:
                    rp_colors[(x,y)] = "#ff0000"
                    
    if len(rp_valid_pts) >= num_travellers:
        rp_base_pts = random.sample(rp_valid_pts, num_travellers)
    else:
        rp_base_pts = rp_valid_pts * (num_travellers // len(rp_valid_pts) + 1)
        rp_base_pts = rp_base_pts[:num_travellers]
        
    print("Processing Code Logo (</>)...")
    # Draw brackets < >
    brackets_canvas = Image.new('1', (300, 340), 0)
    draw_b = ImageDraw.Draw(brackets_canvas)
    draw_b.line([(110, 130), (70, 170), (110, 210)], fill=1, width=12) # <
    draw_b.line([(190, 130), (230, 170), (190, 210)], fill=1, width=12) # >
    
    # Draw slash /
    slash_canvas = Image.new('1', (300, 340), 0)
    draw_s = ImageDraw.Draw(slash_canvas)
    draw_s.line([(170, 120), (130, 220)], fill=1, width=12)            # /
    
    b_pts = get_points(np.array(brackets_canvas), int(num_travellers * 0.66))
    s_pts = get_points(np.array(slash_canvas), num_travellers - len(b_pts))
    
    # Combine points and keep track of colors
    code_base_pts = b_pts + s_pts
    random.shuffle(code_base_pts) # Shuffle so optimal transport is fair
    
    code_colors = {}
    for pt in code_base_pts:
        if pt in b_pts:
            code_colors[pt] = "#ffffff" # Brackets are white
        else:
            code_colors[pt] = "#ff0000" # Slash is red
    
    random_pts = [(random.randint(0, 300), random.randint(0, 340)) for _ in range(num_travellers)]
    
    print("Running optimal transport Phase 1 (Face -> RP)...")
    start_arr = np.array(face_pts)
    rp_arr_pts = np.array(rp_base_pts)
    dist1 = cdist(start_arr, rp_arr_pts)
    _, col1 = linear_sum_assignment(dist1)
    rp_ordered = rp_arr_pts[col1]
    
    print("Running optimal transport Phase 2 (RP -> Code)...")
    code_arr_pts = np.array(code_base_pts)
    dist2 = cdist(rp_ordered, code_arr_pts)
    _, col2 = linear_sum_assignment(dist2)
    code_ordered = code_arr_pts[col2]
    
    print("Building SVG...")
    kt = "0; 0.1; 0.3; 0.4; 0.6; 0.7; 0.9; 1"
    
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 340" width="100%" height="100%">']
    svg.append('<rect width="100%" height="100%" fill="#000000"/>')
    
    num_groups = 50
    static_groups = [[] for _ in range(num_groups)]
    for i, pt in enumerate(static_pts):
        static_groups[i % num_groups].append(f"M{pt[0]},{pt[1]}h1")
        
    for i, g in enumerate(static_groups):
        path_str = " ".join(g)
        dx = random.randint(-15, 15)
        dy = random.randint(-15, 15)
        svg.append(f'<path d="{path_str}" stroke="#ff0000" stroke-width="1.5" shape-rendering="crispEdges">')
        svg.append(f'  <animate attributeName="opacity" values="0; 1; 1; 0; 0; 0; 0; 0" keyTimes="{kt}" dur="14s" repeatCount="indefinite" />')
        svg.append(f'  <animateTransform attributeName="transform" type="translate" values="{dx*5},{dy*5}; 0,0; 0,0; {dx},{dy}; {dx},{dy}; {dx},{dy}; {dx},{dy}; {dx*5},{dy*5}" keyTimes="{kt}" dur="14s" repeatCount="indefinite" />')
        svg.append('</path>')
        
    for i in range(num_travellers):
        rx, ry = random_pts[i]
        fx, fy = face_pts[i]
        px, py = rp_ordered[i]
        cx, cy = code_ordered[i]
        
        rp_c = rp_colors.get(tuple(px), "#ffffff")
        code_c = code_colors.get(tuple(cx), "#ffffff")
        
        svg.append(f'<rect width="2" height="2" fill="#ff0000">')
        svg.append(f'  <animate attributeName="x" values="{rx}; {fx}; {fx}; {px}; {px}; {cx}; {cx}; {rx}" keyTimes="{kt}" dur="14s" repeatCount="indefinite" />')
        svg.append(f'  <animate attributeName="y" values="{ry}; {fy}; {fy}; {py}; {py}; {cy}; {cy}; {ry}" keyTimes="{kt}" dur="14s" repeatCount="indefinite" />')
        svg.append(f'  <animate attributeName="fill" values="#ff0000; #ff0000; #ff0000; {rp_c}; {rp_c}; {code_c}; {code_c}; #ff0000" keyTimes="{kt}" dur="14s" repeatCount="indefinite" />')
        svg.append('</rect>')
        
    svg.append('</svg>')
    
    with open(output_path, 'w') as f:
        f.write("\\n".join(svg))
    print(f"Saved {output_path}")

generate_animated_svg('IMG_2290-without BG.png', 'rp-logo.png', 'dithered-animated.svg')
