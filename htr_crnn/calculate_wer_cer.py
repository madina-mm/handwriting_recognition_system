import os
import jiwer

def calculate_wer_cer(original_dir, prediction_dir):
    # original_files = os.listdir(original_dir)
    prediction_files = os.listdir(prediction_dir)

    total_wer, total_cer = 0.0, 0.0
    num_files = 0

    for file_name in prediction_files:
        with open(os.path.join(original_dir, file_name), 'r', encoding='utf-8') as file:
            original_text = file.read().strip()
        with open(os.path.join(prediction_dir, file_name), 'r', encoding='utf-8') as file:
            prediction_text = file.read().strip()

        wer = jiwer.wer(original_text, prediction_text)
        cer = jiwer.cer(original_text, prediction_text)
        
        total_wer += wer*100
        total_cer += cer*100
        num_files += 1

        if wer>0 or cer>0:

            print(f"\nFile: {file_name} - WER: {wer:.3f}, CER: {cer:.3f}")
            print('Label     : ', original_text)
            print('Prediction: ', prediction_text)

    average_wer = total_wer / num_files
    average_cer = total_cer / num_files

    print(f"Average WER: {average_wer:.3f}")
    print(f"Average CER: {average_cer:.3f}")

# Example usage:
original_dir = 'data/labels'
prediction_dir = 'data/predictions_ep5'
calculate_wer_cer(original_dir, prediction_dir)
