import pandas
import hanja
import yaml

def hanja_cleaning(sent):
    sent = hanja.translate(sent, 'substitution')
    return sent

def japanese_cleaning(sent):
    with open("./jp_to_kor.yaml") as f:
        jp_to_kor = yaml.load(f, Loader=yaml.FullLoader)
    new_sent = ''
    for char in sent:
        try:
            new_sent+=jp_to_kor[char]
        except:
            new_sent+=char
    
    return new_sent
