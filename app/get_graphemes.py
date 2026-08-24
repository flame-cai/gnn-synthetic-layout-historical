import unicodedata


def get_gc_tokens(devanagari_string):

    # Pre-compile set lookups for better performance
    combining_marks = frozenset({
        '_','्', 'ँ', 'ं', 'ः', 'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'ॄ', 'ॢ', 'ॣ', 'े', 'ै','ॆ', 'ो', 'ौ','ॊ', '़','ॅ','ॉ',
        '_', 'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'ॄ', 'ॢ', 'ॣ', 'े', 'ै', 'ो', 'ौ', '़', 'ॅ', 'ॉ', 'ँ','्',
        '̪', 'ː', 'ʱ', 'ᵊ','ʰ', '̃', '̩', '̯', '̤', '̥', '̬', '̰', '̱', '̲', '̳', '̴', '̵', '̶',
        '̷', '̸', '̹', '̺', '̻', '̼', '̽', '̾', '̿', '̀', '́', '͆', '͇', '͈', '͉',
        '͊', '͋', '͌', '͍', '͎', '͐', '͑', '͒', '͓', '͔', '͕', '͖', '͗', '͘', '͙',
        '͚', '͛', '͜', '͝', '͞', '͟', '͠', '͡', '͢', 'ͣ', 'ͤ', 'ͥ', 'ͦ', 'ͧ', 'ͨ',
        'ͩ', 'ͪ', 'ͫ', 'ͬ', 'ͭ', 'ͮ', 'ͯ', '̚', '̣', '̇', '̈', '̊', '̋', '̌', '̍', '̎',
        '̏', '̓', '̔', '̕', '̖', '̗', '̘', '̙', '̜', '̝', '̞', '̟', '̠', '̡', '̢',
        '̦', '̨', 
        # '.'
        # 'ĩː','ɑː',
    })

    digits = frozenset('०१२३४५६७८९0123456789')
    end_markers = frozenset(['্', '्','͡',])
    
    graphemes = []
    temp_grapheme = []
    
    devanagari_string = devanagari_string.replace(" ", "_")
    
    for char in devanagari_string:
        if char not in combining_marks:
            if temp_grapheme and not (temp_grapheme[-1] in end_markers):
                if char in digits:
                    graphemes.append(''.join(temp_grapheme))
                    temp_grapheme = []
                    graphemes.append(char)
                else:
                    graphemes.append(''.join(temp_grapheme))
                    temp_grapheme = [char]
            else:
                temp_grapheme.append(char)
        else:
            temp_grapheme.append(char)
    
    if temp_grapheme:
        graphemes.append(''.join(temp_grapheme))
    
    # Process all cleanups in one pass
    if '\u200c' in devanagari_string or '\u200d' in devanagari_string or '\u200b' in devanagari_string:
        graphemes = [g.replace('\u200c', '').replace('\u200d', '').replace('\u200b', '') for g in graphemes]
    
    return graphemes




