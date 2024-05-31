from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import re
import os
import json

input_dir = Path("data/mnt")  
fonts_dir = "data/fonts"
fonts_file_path = 'fonts.json'

output_dir = Path("data/images7")
output_dir.mkdir(exist_ok=True)

labels_dir = Path("data/text_labels7")
labels_dir.mkdir(exist_ok=True) 


with open(fonts_file_path, 'r') as fonts_file:
    fonts_data = json.load(fonts_file)

sentence_counter = 0
max_sentences = 23285

symbols = '''.,;:?!­/()[]"”“'…«»-$%@&–+—`'''

def text_to_image(text, sentence_counter, fonts_data, fonts_dir, output_path, labels_dir, symbols):
    
    # Split text into sentences
    parts  = re.split(r'(?<!\d)(?<![A-ZİÖÜĞƏÇŞ])([.!?]) +', text)
    
    # Reassemble sentences with their delimiters
    sentences = [parts[i] + (parts[i+1] if i+1 < len(parts) else '') for i in range(0, len(parts), 2)]
    
    # Process every sentence
    for i, sentence in enumerate(sentences):
        if sentence_counter >= max_sentences:  # Stop if max reached
            return sentence_counter  # Return the current count
        
        font_number = sentence_counter % len(fonts_data)
        # if font_number != 7:
        #     sentence_counter += 1
        #     continue
        font_data = fonts_data[str(font_number)]
        font_name = font_data['font_name']
        font_size = font_data['font_size']
        font_width_factor = font_data['font_width_factor']
        line_spacing = font_data['line_spacing']

        font_path = os.path.join(fonts_dir, font_name)
        font = ImageFont.truetype(str(font_path), font_size)
       
        # Simplified approach for demonstration: split long text into lines
        lines = []
        cleaned_sentence = ''.join(char for char in sentence if char not in symbols)
        words = cleaned_sentence.split()
        if len(words)<5:
            sentence_counter += 1
            continue
        
        max_width = 1000
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
            image = Image.new('RGB', (max_width, img_height), 'white')
            draw = ImageDraw.Draw(image)

            draw.text((left_margin, top_margin), line.strip(), fill="black", font=font)
            
            # Save the line as an image
            line_img_output_path = str(output_path) + f'_{i}_{j}_{font_name[:-4]}.png'
            image.save(line_img_output_path)
            print(f"Image saved: {line_img_output_path}")

            # Save the label for the line

            line_filename = os.path.basename(line_img_output_path).replace('.png', '.txt')
            line_output_path = f'{labels_dir}\{line_filename}'

            with open(line_output_path, 'w', encoding='utf-8') as f:
                f.write(line.strip())
            print(f"Text label saved: {line_output_path}")

        # Increment sentence counter after processing all lines in the sentence
        sentence_counter += 1
        print(sentence_counter, '\n\n')

    return sentence_counter

# Processing
for txt_file in input_dir.glob("*.txt"):
    if sentence_counter >= max_sentences:  # Check before processing a new file
        break
    with txt_file.open('r', encoding='utf-8') as file:
        content = file.read().strip()
        output_file = output_dir / txt_file.stem
        sentence_counter = text_to_image(content, sentence_counter, fonts_data, fonts_dir, output_file, labels_dir, symbols)