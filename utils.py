import numpy as np
import pandas as pd

def calculate_avg_height_and_midpoints(df):
    heights = []
    midpoints = []
    
    for _, row in df.iterrows():
        coordinates = list(map(int, row['coordinates'].split(',')))
        x_min, y_min, x_max, y_max = coordinates[0], coordinates[1], coordinates[2], coordinates[3]
        
        height = y_max - y_min
        midpoint = (y_min + y_max) / 2
        heights.append(height)
        midpoints.append(midpoint)
    
    avg_height = np.mean(heights)
    return avg_height, midpoints

def group_words_by_flexible_lines(df, avg_height, midpoints):
    lines = []
    
    for i, row in df.iterrows():
        # Extract word coordinates and midpoint
        coordinates = list(map(int, row['coordinates'].split(',')))
        x_min, y_min, x_max, y_max = coordinates[0], coordinates[1], coordinates[2], coordinates[3]
        
        midpoint = midpoints[i]
        
        # Find or create a line based on vertical proximity (within a threshold of avg height)
        added_to_line = False
        for line in lines:
            line_midpoint = np.mean([word['midpoint'] for word in line])  # Average of midpoints in the line
            if abs(midpoint - line_midpoint) < avg_height * 0.6:  # Threshold set as 60% of average height
                line.append({'prediction': row['prediction'], 'coordinates': (x_min, y_min, x_max, y_max), 'midpoint': midpoint})
                added_to_line = True
                break
        
        # If no line found, create a new line
        if not added_to_line:
            lines.append([{'prediction': row['prediction'], 'coordinates': (x_min, y_min, x_max, y_max), 'midpoint': midpoint}])

    return lines

def sort_lines_and_words(lines):
    # Sort words within each line by their x_min (horizontal position)
    for line in lines:
        line.sort(key=lambda word: word['coordinates'][0])  # Sort by x_min

    # Sort lines themselves by their average midpoint (top to bottom order)
    lines.sort(key=lambda line: np.mean([word['midpoint'] for word in line]))  # Sort by average midpoint

    return lines

def reconstruct_paragraph(lines):
    paragraph = []
    for line in lines:
        line_text = ' '.join([word['prediction'] for word in line])  # Join the words in each line
        paragraph.append(line_text)
    return '\n'.join(paragraph)