import unicodedata

def list_correct_grapheme_clusters(devanagari_string):
    combining_marks = {'्', 'ँ', 'ं', 'ः', 'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'ॄ', 'ॢ', 'ॣ', 'े', 'ै', 'ो', 'ौ', '़','ॅ'}
    graphemes = []
    temp_grapheme = ""
    for char in devanagari_string:
        if unicodedata.combining(char) == 0 and char not in combining_marks:
            # If temp_grapheme has something and last char was not virama, append to graphemes
            if temp_grapheme and not temp_grapheme.endswith('्'):
                graphemes.append(temp_grapheme)
                temp_grapheme = char
            else:
                temp_grapheme += char
        else:
            temp_grapheme += char  # Add combining mark or virama to current grapheme

    # Append the last grapheme if exists
    if temp_grapheme:
        graphemes.append(temp_grapheme)

    return graphemes