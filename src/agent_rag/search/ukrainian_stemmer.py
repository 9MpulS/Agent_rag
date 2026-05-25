import re

class UkrainianStemmer:
    """
    Ukrainian Porter Stemmer.
    Based on standard stemming rules for Ukrainian.
    """
    vowel = r'аеиоуюяіїє'
    perfectiveground = r'(ив|ивши|ившись|ыв|ывши|ывшись((?<=[ая])(в|вши|вшись)))$'
    reflexive = r'(с[яьи])$'
    adjective = r'(ими|ій|ий|а|е|ова|ове|ів|є|їй|єє|еє|я|ім|ем|им|ім|их|іх|ою|йми|іми|у|ю|ого|ому|ої)$'
    participle = r'(ий|ого|ому|им|ім|а|ій|у|ою|ій|і|их|йми|их)$'
    verb = r'(сь|ся|ив|ать|ять|у|ю|ав|али|учи|ячи|вши|ши|е|ме|ати|яти|є)$'
    noun = r'(а|ев|ов|е|ями|ами|еи|и|ей|ой|ий|й|иям|ям|ием|ем|ам|ом|о|у|ах|иях|ях|ы|ь|ию|ью|ю|ия|ья|я|і|ові|ї|ею|єєю|ою|є|еві|ем|єм|ів|їв|ю)$'
    rvre = re.compile(f'[^[{vowel}]]*[{vowel}]+(.*)', re.IGNORECASE)
    derivational = r'[^аеиоуюяіїє][аеиоуюяіїє]+[^аеиоуюяіїє]+[аеиоуюяіїє].*(?<=о)сть?$'

    @staticmethod
    def s(st, reg, to):
        orig = st
        st = re.sub(reg, to, st)
        return orig != st, st

    @classmethod
    def stem_word(cls, word):
        word = word.lower()
        word = word.replace('ё', 'е')
        
        match = cls.rvre.search(word)
        if not match:
            return word
            
        start = match.start(1)
        rv = word[start:]
        
        # Step 1
        is_perfective, rv = cls.s(rv, cls.perfectiveground, '')
        if not is_perfective:
            _, rv = cls.s(rv, cls.reflexive, '')
            is_adj, rv = cls.s(rv, cls.adjective, '')
            if is_adj:
                cls.s(rv, cls.participle, '')
            else:
                is_verb, rv = cls.s(rv, cls.verb, '')
                if not is_verb:
                    cls.s(rv, cls.noun, '')
        
        # Step 2
        _, rv = cls.s(rv, 'и$', '')
        
        # Step 3
        if re.search(cls.derivational, rv):
            _, rv = cls.s(rv, 'ость$', '')
            
        # Step 4
        is_i, rv = cls.s(rv, 'ь$', '')
        if not is_i:
            _, rv = cls.s(rv, 'ейше?', '')
            _, rv = cls.s(rv, 'нн$', 'н')
            
        return word[:start] + rv

def stem_ukrainian_text(text: str) -> str:
    """Stems a given text containing Ukrainian words, preserving punctuation."""
    if not text:
        return text
    
    def replace_word(match):
        return UkrainianStemmer.stem_word(match.group(0))
        
    return re.sub(r'[а-яА-ЯїієґЇІЄҐa-zA-Z]+', replace_word, text)
