import re
import string
FORMAL_DIGIT="零一二三四五六七八九十百千万亿"
LARGE_FORMAL_DIGIT="零壹贰叁肆伍陆柒捌玖拾佰仟萬億"
math_digit="1234567890\uFF10\uFF11\uFF12\uFF13\uFF14\uFF15\uFF16\uFF17\uFF18\uFF19"
number_digit = "n1234567890.a-zA-Z"
punctuation = "×÷．！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰–‘’‛“”„‟…‧﹏>"
pattern_formal = re.compile('['+FORMAL_DIGIT+']+')
pattern_large = re.compile('['+ LARGE_FORMAL_DIGIT +']+')
pattern_digit = re.compile('['+math_digit+']+')
pattern_number = re.compile('['+number_digit+']+')
pattern_punctuation = re.compile('['+punctuation+string.punctuation+']+')
result = []
with open('vocab.txt','r',encoding='utf-8') as f:
    for line in f:
        if pattern_formal.match(line.strip()):
            continue
        if pattern_large.match(line.strip()):
            continue
        if pattern_digit.match(line.strip()):
            continue
        if pattern_number.match(line.strip()):
            continue
        if pattern_punctuation.match(line.strip()):
            continue
        result.append(line.strip())

with open('vocab_trim.txt','w', encoding='utf-8') as f:
    for line in result:
        f.write(line+'\n')
print(len(result))
print('FIN')
