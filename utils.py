import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

def calculate_avg_height_and_midpoints(df: pd.DataFrame) -> Tuple[float, List[float]]:
    heights = []
    midpoints = []
    
    for _, row in df.iterrows():
        coordinates = list(map(int, row['coordinates'].split(',')))
        x_min, y_min, x_max, y_max = coordinates[0], coordinates[1], coordinates[2], coordinates[3]
        
        height = y_max - y_min
        midpoint = (y_min + y_max) / 2
        heights.append(height)
        midpoints.append(midpoint)
    
    # Use median instead of mean to be more robust to outliers
    avg_height = np.median(heights)
    return avg_height, midpoints

def group_words_by_flexible_lines(df: pd.DataFrame, avg_height: float, midpoints: List[float]) -> List[List[Dict]]:
    lines = []
    processed_indices = set()
    
    # Sort words by y-coordinate (top to bottom)
    sorted_indices = sorted(range(len(midpoints)), key=lambda i: midpoints[i])
    
    for i in sorted_indices:
        if i in processed_indices:
            continue
            
        current_midpoint = midpoints[i]
        current_line = []
        
        # Get coordinates for the current word
        coords_i = list(map(int, df.iloc[i]['coordinates'].split(',')))
        y_min_i, y_max_i = coords_i[1], coords_i[3]
        
        # Add the current word to the line
        current_line.append({
            'prediction': df.iloc[i]['prediction'],
            'coordinates': tuple(coords_i),
            'midpoint': current_midpoint
        })
        processed_indices.add(i)
        
        # Look for words that belong to the same line
        for j in range(len(midpoints)):
            if j in processed_indices:
                continue
                
            coords_j = list(map(int, df.iloc[j]['coordinates'].split(',')))
            y_min_j, y_max_j = coords_j[1], coords_j[3]
            
            # Calculate vertical overlap
            overlap = min(y_max_i, y_max_j) - max(y_min_i, y_min_j)
            min_height = min(y_max_i - y_min_i, y_max_j - y_min_j)
            overlap_ratio = overlap / min_height if min_height > 0 else 0
            
            # Check if words belong to the same line using multiple criteria
            vertical_distance = abs(midpoints[j] - current_midpoint)
            if (vertical_distance < avg_height * 0.5) or (overlap_ratio > 0.5):
                current_line.append({
                    'prediction': df.iloc[j]['prediction'],
                    'coordinates': tuple(coords_j),
                    'midpoint': midpoints[j]
                })
                processed_indices.add(j)
        
        lines.append(current_line)
    
    return lines

def sort_lines_and_words(lines: List[List[Dict]]) -> List[List[Dict]]:
    # Sort words within each line by x-coordinate
    for line in lines:
        line.sort(key=lambda word: word['coordinates'][0])
    
    # Sort lines by average y-coordinate
    lines.sort(key=lambda line: np.mean([word['midpoint'] for word in line]))
    
    return lines

def reconstruct_paragraph(lines):
    paragraph = []
    for line in lines:
        line_text = ' '.join([word['prediction'] for word in line])  # Join the words in each line
        paragraph.append(line_text)
    return '\n'.join(paragraph)