from recognition.utils import *
import pandas as pd
import regex as re


def non_matching_graphemes(s, s1):

    # Split both strings into graphemes
    graphemes_s = list_correct_grapheme_clusters(s)
    graphemes_s1 = list_correct_grapheme_clusters(s1)

    # Find the LCS to align graphemes instead of characters
    m, n = len(graphemes_s), len(graphemes_s1)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if graphemes_s[i - 1] == graphemes_s1[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Reconstruct the sequence of non-matching graphemes
    non_matching = []
    error_count = 0
    while m > 0 and n > 0:
      # if the grapheme clusters match
        if graphemes_s[m - 1] == graphemes_s1[n - 1]:
            m, n = m - 1, n - 1

      # if they dont match
        else:
            temp_s, temp_s1 = "", ""

            while m > 0 and (n == 0 or dp[m][n] == dp[m-1][n]):
                temp_s = graphemes_s[m - 1] + temp_s
                m -= 1

            while n > 0 and (m == 0 or dp[m][n] == dp[m][n-1]):
                temp_s1 = graphemes_s1[n - 1] + temp_s1
                n -= 1

            # if(len(temp_s)>3):
              # print(list_correct_grapheme_clusters(temp_s))
            if(len(list_correct_grapheme_clusters(temp_s))>=len(list_correct_grapheme_clusters(temp_s1))):

              temp_list = list_correct_grapheme_clusters(temp_s)

            else:
              temp_list = list_correct_grapheme_clusters(temp_s1)

            temp_list.reverse()
            error_count += len(temp_list)
            non_matching.extend(temp_list)

    non_matching.reverse()
    
    return(error_count/len(graphemes_s))

def longest_common_subsequence(s, s1):
    # Create the DP table
    m, n = len(s), len(s1)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s[i - 1] == s1[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Reconstruct the LCS
    lcs = []
    while m > 0 and n > 0:
        if s[m - 1] == s1[n - 1]:
            lcs.append(s[m - 1])
            m -= 1
            n -= 1
        elif dp[m - 1][n] > dp[m][n - 1]:
            m -= 1
        else:
            n -= 1

    # The LCS is built backwards, so reverse it
    return ''.join(reversed(lcs))

def get_unicode_eval(pg_annot,pg_pred):
  s = pg_pred
  s1 = pg_annot

  lcs = longest_common_subsequence(s, s1)
#   print(lcs)
  # print(modelname+" LCS: "+f'{len(lcs)/len(s)}')
  denom = len(s1) if len(s1)>= len(s) else len(s)
  return(1-(len(lcs)/denom))

def get_error(pg_annot,pg_pred):

  errors = non_matching_graphemes(pg_annot, pg_pred)

  return(errors)

def get_txt(model, doc, folder):
  
  folder_data = {'val_manuscript1':[i for i in range(3,8,2)],
                 'test_manuscript1':[i for i in range(9,18,2)], #range(47,52) or #range(9,18,2)
                }

  model_data = {'manuscript1':{'file_num':3976}}

  path = f'non_easyocr_predictions/{model}/{doc}'

  txt = ''
  for num_page in folder_data[f'{folder}_{doc}']:

    file_num = model_data[doc]['file_num']
    # tesseract ocr prediction is tess_predictions
    filename = f'tess_{file_num}_{num_page:04d}.txt' if model=='tess_predictions/page_level_predictions' else f'{file_num}_{num_page:04d}.txt'
    with open(f"{path}/{filename}", 'rt') as myfile:
        txt += myfile.read()

  return(txt)

# evaluates the models using grapheme cluster error rate
def run_eval(doc):
  
#   commented put code is related to other ocr models for the evaluation comparison

  error_list = {'val':[],
                'test':[]}

  # model_list = [f'{doc}_{i:02d}' for i in range(1,8)]
  model_list = [f'{doc}_{i:02d}' for i in [7,51]]

  # path to annotated images
  path_annot = 'recognition/line_images'

  # print heading
  print('folder/model', end='\t')
#   print('EasyOCR', end='\t\t')
#   for i in model_list:
#     print(i, end='\t\t')
#   print('DocAI\t\tTesseract')
  print('\n')

  # iterate over val and test folders
  for i in ['test','val']:
    print(f'{i}', end='\t\t')

    # get annotated page text
    pg_annot = f'{i}_{doc}/labels.txt'  # example: val_manuscript1 or test_manuscript1
    annot_text = pd.read_table(f'{path_annot}/{pg_annot}', encoding="utf-8",header=None)
    annot_text.columns = ['image_path','labels']
    annot_text = annot_text.labels.str.cat(sep='').replace(' ','') #CHANGED

    # print(annot_text)

    # # to get vanilla easyocr's prediction and error rate
    # pred_text_veo = get_txt('vanillaeasyocr_predictions/page_level_predictions',doc,folder=i).replace('\n','')
    # error_list[i].append(get_error(annot_text,pred_text_veo))
    # print(f'{get_error(annot_text,pred_text_veo):.3f}', end='\t\t')
    
    # print(pred_text_veo)
    # iterate over each model for test/val folder
    for model_name in model_list:

      # get predicted text for this model
      pred_text_path = f'recogntion/line_pred/{i}_{doc}_{model_name}.txt'  #example: val_manuscript1_manuscript1_12
      with open(pred_text_path, 'rt') as myfile:
        pred_text = myfile.read().replace('\n','')

      # get error rate
      error_list[i].append(get_error(annot_text,pred_text))
      print(f'{get_error(annot_text,pred_text):.3f}', end='\t\t') 

    
    # # get google's prediction and error rate
    # pred_text_google = get_txt('document_ai_predictions/page_level_predictions',doc,folder=i).replace('\n','')
    # print(f'{get_error(annot_text,pred_text_google):.3f}', end='\t\t')

    # # get tesseract's prediction and error rate
    # pred_text_tess = get_txt('tess_predictions/page_level_predictions',doc,folder=i).replace('\n','')
    # print(f'{get_error(annot_text,pred_text_tess):.3f}', end='\t\t\t')

    print('\n')
  return(error_list)

def run_unicode_eval(doc):

  error_list = {'val':[],
                'test':[]}

  # model_list = [f'{doc}_{i:02d}' for i in range(1,8)]
  model_list = [f'{doc}_{i:02d}' for i in [7,51]]

  # path to annotated images
  path_annot = 'recognition/line_images'

  # print heading
  print('folder/model', end='\t')
#   print('EasyOCR', end='\t\t')
#   for i in model_list:
#     print(i, end='\t\t')
#   print('DocAI\t\tTesseract')
  print('\n')

  # iterate over val and test folders
  for i in ['test','val']:
    print(f'{i}', end='\t\t')

    # get annotated page text
    pg_annot = f'{i}_{doc}/labels.txt'  # example: val_manuscript1 or test_manuscript1
    annot_text = pd.read_table(f'{path_annot}/{pg_annot}', encoding="utf-8",header=None)
    annot_text.columns = ['image_path','predicted_labels']
    annot_text = annot_text.predicted_labels.str.cat(sep='').replace(' ','') #CHANGED

    # print(annot_text)

    # # get vanilla easyocr's prediction and error rate
    # pred_text_veo = get_txt('vanillaeasyocr_predictions/page_level_predictions',doc,folder=i).replace('\n','')
    # error_list[i].append(get_unicode_eval(annot_text,pred_text_veo))
    # print(f'{get_unicode_eval(annot_text,pred_text_veo):.3f}', end='\t\t')
    
    # print(pred_text_veo)
    # iterate over each model for test/val folder
    for model_name in model_list:

      # get predicted text for this model
      pred_text_path = f'recognition/line_pred/{i}_{doc}_{model_name}.txt'  #example: val_manuscript1_manuscript1_12
      with open(pred_text_path, 'rt') as myfile:
        pred_text = myfile.read().replace('\n','')

      # get error rate
      error_list[i].append(get_unicode_eval(annot_text,pred_text))
      print(f'{get_unicode_eval(annot_text,pred_text):.3f}', end='\t\t') 

    
    # # get google's prediction and error rate
    # pred_text_google = get_txt('document_ai_predictions/page_level_predictions',doc,folder=i).replace('\n','')
    # print(f'{get_unicode_eval(annot_text,pred_text_google):.3f}', end='\t\t')

    # # get tesseract's prediction and error rate
    # pred_text_tess = get_txt('tess_predictions/page_level_predictions',doc,folder=i).replace('\n','')
    # print(f'{get_unicode_eval(annot_text,pred_text_tess):.3f}', end='\t\t\t')

    print('\n')
  return(error_list)

# Example usage

run_eval('manuscript1')
run_unicode_eval('manuscript1')