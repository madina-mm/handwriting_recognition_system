import os
import jiwer 

def print_predictions(original_dir, prediction_dirs):

    for file_name in os.listdir(prediction_dirs[0]):
        with open(os.path.join(original_dir, file_name), 'r', encoding='utf-8') as file:
            original_text = file.read().strip()
        
        # print('\nText: ', original_text)
        WERS = []
        CERS = []   
        for dir in prediction_dirs:
            with open(os.path.join(dir, file_name), 'r', encoding='utf-8') as file:
                prediction_text = file.read().strip()

            wer = jiwer.wer(original_text, prediction_text)
            cer = jiwer.cer(original_text, prediction_text)

            if wer>0:
                WERS.append(wer)
            
            if cer>0:
                CERS.append(wer)

        if len(WERS)==2 or len(CERS)==2:
            print('\n', file_name)
            print('\nLabel   :', original_text)

            for dir in prediction_dirs:
                with open(os.path.join(dir, file_name), 'r', encoding='utf-8') as file:
                    prediction_text = file.read().strip()

                wer = jiwer.wer(original_text, prediction_text)
                cer = jiwer.cer(original_text, prediction_text)

                if wer>0 or cer>0:
                    if dir == 'data/predictions_first':
                        print(f'Initial : {prediction_text}')
                    else:
                        print(f'Epoch {dir[-1]} : {prediction_text}')


                elif dir == 'data/predictions_ep2':
                    print(f'Epoch {dir[-1]} : {prediction_text}')


                




original_dir = 'data/labels'
predictions_dir_demo = 'data/predictions_first'
prediction_dir1 = 'data/predictions_ep1'
prediction_dir2 = 'data/predictions_ep2'
# prediction_dir3 = 'data/predictions_ep3'
# prediction_dir5 = 'data/predictions_ep5'

print_predictions(original_dir, [predictions_dir_demo, prediction_dir1, prediction_dir2])