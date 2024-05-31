from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import re
import os
import json

fonts_dir = "data/fonts"
fonts_file_path = 'fonts.json'

output_dir = Path("data/images_font_samples")
output_dir.mkdir(exist_ok=True)

with open(fonts_file_path, 'r') as fonts_file:
    fonts_data = json.load(fonts_file)


symbols = '''.,;:?!­/()[]"”“'…«»-$%@&–+—`'''

def text_to_image(text, fonts_data, fonts_dir, output_path, symbols):
    
    # Split text into sentences
    parts  = re.split(r'(?<!\d)(?<![A-ZİÖÜĞƏÇŞ])([.!?]) +', text)
    
    # Reassemble sentences with their delimiters
    sentences = [parts[i] + (parts[i+1] if i+1 < len(parts) else '') for i in range(0, len(parts), 2)]
    
    # Process every sentence
    for i, sentence in enumerate(sentences):
        
        for font_number in range(len(fonts_data)):

            font_data = fonts_data[str(font_number)]
            font_name = font_data['font_name']
            font_size = font_data['font_size']+30
            font_width_factor = font_data['font_width_factor']
            line_spacing = font_data['line_spacing']

            font_path = os.path.join(fonts_dir, font_name)
            font = ImageFont.truetype(str(font_path), font_size)
        
            # Simplified approach for demonstration: split long text into lines
            lines = []
            cleaned_sentence = ''.join(char for char in sentence if char not in symbols)
            words = cleaned_sentence.split()

            
            max_width = 500
            left_margin = 35
            top_margin = 30
            line = ''
            for word in words:
                if len(line + word) < max_width // (font_size * font_width_factor):
                    line += (word + ' ')
                else:
                    lines.append(line)
                    line = word + ' '
            lines.append(line)  # add the last line

            # Draw text
            for j, line in enumerate(lines):
                # Create an image for each line
                line_height = font_size + line_spacing
                img_height = line_height + 2*top_margin
                image = Image.new('RGB', (max_width, img_height), 'black')
                draw = ImageDraw.Draw(image)

                draw.text((left_margin, top_margin), line.strip(), fill="white", font=font)
                
                # Save the line as an image
                line_img_output_path = str(output_path) + f'_{i}_{j}_{font_name[:-4]}.png'
                image.save(line_img_output_path)
                print(f"Image saved: {line_img_output_path}")


# Processing
content = 'Nümunə'
output_file = output_dir / 'sample'
text_to_image(content, fonts_data, fonts_dir, output_file, symbols)