import numpy as np

def median_cut_quantization(image, num_colors):
    image = np.array(image)
    pixels = image.reshape(-1, 3)
    
    # Création de la palette de couleurs
    palette = np.zeros((num_colors, 3), dtype=np.uint8)
    
    # Sélection des couleurs initiales pour la palette
    initial_colors = select_initial_colors(pixels, num_colors)
    
    # Appliquer l'algorithme Median Cut
    split_palette(pixels, palette, initial_colors, 0)
    
    # Conversion des pixels de l'image vers les couleurs de la palette
    quantized_image = apply_palette(image, palette)
    
    return quantized_image

def select_initial_colors(pixels, num_colors):
    initial_colors = np.zeros((num_colors, 3), dtype=np.uint8)
    
    # Sélectionner les couleurs initiales en fonction des valeurs extrêmes des canaux RGB
    for i in range(3):
        min_val = np.min(pixels[:, i])
        max_val = np.max(pixels[:, i])
        initial_colors[:, i] = np.linspace(min_val, max_val, num_colors)
    
    return initial_colors

def split_palette(pixels, palette, colors, level):
    if level >= len(colors):
        return
    
    indices = np.where(np.all(pixels == colors[level], axis=1))[0]
    median = np.median(indices)
    palette[level] = colors[level]
    
    split_palette(pixels[:median], palette, colors, level + 1)
    split_palette(pixels[median:], palette, colors, level + 1)

def apply_palette(image, palette):
    h, w, _ = image.shape
    quantized_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    for i in range(h):
        for j in range(w):
            pixel = image[i, j]
            closest_color = find_closest_color(pixel, palette)
            quantized_image[i, j] = closest_color
    
    return quantized_image

def find_closest_color(pixel, palette):
    distances = np.linalg.norm(palette - pixel, axis=1)
    closest_index = np.argmin(distances)
    closest_color = palette[closest_index]