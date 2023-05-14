import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from copy import copy
import json
import numpy as np
import random
import pickle
import re
from typing import List, Optional, Tuple

wordnet = {}
with open("./src/augmentation/wordnet.pickle", "rb") as f:
	wordnet = pickle.load(f)

# 한글만 남기고 나머지는 삭제
def get_only_hangul(line):
	parseText= re.compile('/ ^[ㄱ-ㅎㅏ-ㅣ가-힣@#]*$/').sub('',line)
	return parseText

########################################################################
# Synonym replacement
# Replace n words in the sentence with synonyms from wordnet
########################################################################
def synonym_replacement(words: List[str], n: int)->List[str]:
	new_words = words.copy()

	random_word_list=[]
	for word in words:
		if '#' in word or '@' in word:
			continue
		random_word_list.append(word)
	random_word_list = list(set(random_word_list))
	# print(random_word_list)
	random.shuffle(random_word_list)

	num_replaced = 0
	for random_word in random_word_list:
		synonyms = get_synonyms(random_word)
		if len(synonyms) >= 1:
			synonym = random.choice(list(synonyms))
			new_words = [synonym if word == random_word else word for word in new_words]
			num_replaced += 1
		if num_replaced >= n:
			break

	if len(new_words) != 0:
		sentence = ' '.join(new_words)
		new_words = sentence.split(" ")
	else:
		new_words = ""

	return new_words


def get_synonyms(word):
	synomyms = []

	try:
		for syn in wordnet[word]:
			for s in syn:
				synomyms.append(s)
	except:
		pass

	return synomyms


########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################
def random_deletion(words: List[str], p: float):
	if len(words) == 1:
		return words

	new_words = []
	indices=[]
	for i, word in enumerate(words):
		r = random.uniform(0, 1)
		if '@' in word or '#' in word:
			# 2가지 word는 꼭 들어가도록 보장됨
			new_words.append(word)
		elif r > p:
			new_words.append(word)
			indices.append(i)

	if len(new_words) == len(words):
		new_words.pop(random.choice(indices))

	return new_words

########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################
def random_swap(words, n):
	new_words = words.copy()
	for _ in range(n):
		new_words = swap_word(new_words)

	return new_words

def swap_word(new_words:List[str]):
	random_idx_1 = random.randint(0, len(new_words)-1)
	random_idx_2 = random_idx_1
	counter = 0

	while random_idx_2 == random_idx_1:
		random_idx_2 = random.randint(0, len(new_words)-1)
		counter += 1
		if counter > 3:
			return new_words

	new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
	return new_words


########################################################################
# Random insertion
# Randomly insert n words into the sentence
########################################################################
def random_insertion(words, n):
	new_words = words.copy()
	for _ in range(n):
		add_word(new_words)
	
	return new_words


def add_word(new_words):
	synonyms = []
	counter = 0
	while len(synonyms) < 1:
		if len(new_words) >= 1:
			random_word = new_words[random.randint(0, len(new_words)-1)]
			synonyms = get_synonyms(random_word)
			counter += 1
		else:
			random_word = ""

		if counter >= 10:
			return
		
	random_synonym = synonyms[0]
	random_idx = random.randint(0, len(new_words)-1)
	new_words.insert(random_idx, random_synonym)

def restore(aug_s:str,subj:str,obj:str)->Tuple[str,int,int,int,int]:
	sub_pos=aug_s.find('@')
	obj_pos=aug_s.find('#')
	if sub_pos<obj_pos:
		aug_s.replace('@',subj)
		subject_start=sub_pos
		subject_end=sub_pos+len(subj)-1
		obj_pos=aug_s.find('#')
		aug_s.replace('#',obj)
		object_start=obj_pos
		object_end=obj_pos+len(obj)-1
	else:
		aug_s.replace('#',obj)
		object_start=obj_pos
		object_end=obj_pos+len(obj)-1
		sub_pos=aug_s.find('@')
		aug_s.replace('@',subj)
		subject_start=sub_pos
		subject_end=sub_pos+len(subj)-1
	return aug_s, subject_start, subject_end, object_start, object_end

def EDA(
		sentence:str, 
		subj:str,
		obj:str,
		subject_start:int,
		subject_end:int,
		object_start:int,
		object_end:int,
		num_aug:int,
		rd:bool=True,
		rs:bool=True,
		sr:bool=False,
		ri:bool=False,
		p_rd:float=0.1, 
		alpha_rs:float=0.1, 
		alpha_sr:float=0.1, 
		alpha_ri:float=0.1, 
		shuffle:bool=True,
	)->List[str]:
	# sentence = get_only_hangul(sentence)
	masked_sentence = copy(sentence)
	# subject와 object를 각각 @,#로 마스킹
	if object_start < subject_start:
		if subject_end==len(masked_sentence)-1:
			masked_sentence = masked_sentence[:subject_start]+'@'
		else:
			masked_sentence = masked_sentence[:subject_start]+'@'+masked_sentence[subject_end+1:]
		masked_sentence = masked_sentence[:object_start]+'#'+masked_sentence[object_end+1:]
	else:
		if object_end==len(masked_sentence)-1:
			masked_sentence = masked_sentence[:object_start]+'#'
		else:
			masked_sentence = masked_sentence[:object_start]+'#'+masked_sentence[object_end+1:]
		masked_sentence = masked_sentence[:subject_start]+'@'+masked_sentence[subject_end+1:]

	words = masked_sentence.split(' ')
	words = [word for word in words if word != ""]
	num_words = len(words)

	augmented_sentences = [sentence]
	sub_starts=[subject_start]
	sub_ends=[subject_end]
	obj_starts=[object_start]
	obj_ends=[object_end]

	num_new_per_technique = int(num_aug/float(sr+ri+rd+rs)) + 1


	# sr
	n_sr = max(1, int(alpha_sr*num_words))
	if sr:
		for _ in range(num_new_per_technique):
			a_words = synonym_replacement(words, n_sr)
			aug_s=' '.join(a_words)
			aug_s, subject_start, subject_end, object_start, object_end = restore(aug_s=aug_s,subj=subj, obj=obj)
			sub_starts.append(subject_start)
			sub_ends.append(subject_end)
			obj_starts.append(object_start)
			obj_ends.append(object_end)
			augmented_sentences.append(aug_s)

	# ri
	n_ri = max(1, int(alpha_ri*num_words))
	if ri:
		for _ in range(num_new_per_technique):
			a_words = random_insertion(words, n_ri)
			aug_s=' '.join(a_words)
			aug_s, subject_start, subject_end, object_start, object_end = restore(aug_s=aug_s,subj=subj, obj=obj)
			sub_starts.append(subject_start)
			sub_ends.append(subject_end)
			obj_starts.append(object_start)
			obj_ends.append(object_end)
			augmented_sentences.append(aug_s)

	# rs
	n_rs = max(1, int(alpha_rs*num_words))
	if rs:
		for _ in range(num_new_per_technique):
			a_words = random_swap(words, n_rs)
			aug_s=' '.join(a_words)
			aug_s, subject_start, subject_end, object_start, object_end = restore(aug_s=aug_s,subj=subj, obj=obj)
			sub_starts.append(subject_start)
			sub_ends.append(subject_end)
			obj_starts.append(object_start)
			obj_ends.append(object_end)
			augmented_sentences.append(aug_s)

	# rd
	if rd:
		for _ in range(num_new_per_technique):
			a_words = random_deletion(words, p_rd)
			aug_s=' '.join(a_words)
			aug_s, subject_start, subject_end, object_start, object_end = restore(aug_s=aug_s,subj=subj, obj=obj)
			sub_starts.append(subject_start)
			sub_ends.append(subject_end)
			obj_starts.append(object_start)
			obj_ends.append(object_end)
			augmented_sentences.append(aug_s)

	# augmented_sentences = [get_only_hangul(sentence) for sentence in augmented_sentences]

	if shuffle:
		random.shuffle(augmented_sentences)

	if num_aug >= 1:
		augmented_sentences = augmented_sentences[:num_aug]
	else:
		keep_prob = num_aug / len(augmented_sentences)
		augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

	return augmented_sentences, sub_starts, sub_ends, obj_starts, obj_ends

def EDA_DataFrame(
		df:pd.DataFrame, 
		aug_per_label:int=1000,
		rd:bool=True,
		rs:bool=True,
		sr:bool=False,
		ri:bool=False,
	):
	value_count = df['label'].value_counts()
	new_df = pd.DataFrame(columns=df.columns)
	for label in df['label'].unique():
		rows_per_label=df[df['label']==label].copy()
		if value_count[label]<aug_per_label:
			# label 별로 EDA 수행
			aug_num = aug_per_label//len(rows_per_label)+1
			augmented_df_per_label=pd.DataFrame(columns=rows_per_label.columns)
			for i,row in rows_per_label.iterrows():
				EDA_result = EDA(
					row['sentence'],
					subj=row['subject_entity']['word'],
					obj=row['object_entity']['word'], 
					subject_start=row['subject_entity']['start_idx'],
					subject_end=row['subject_entity']['end_idx'],
					object_start=row['object_entity']['start_idx'],
					object_end=row['object_entity']['end_idx'],
					num_aug=aug_num,
					rd=rd,
					rs=rs,
					sr=sr,
					ri=ri,
				)
				augmented_sentences, sub_starts, sub_ends, obj_starts, obj_ends = EDA_result
				for aug_s, sub_start, sub_end, obj_start, obj_end in zip(augmented_sentences, sub_starts, sub_ends, obj_starts, obj_ends):
					new_row=row.copy()
					new_row['sentence']=aug_s
					new_row['subject_entity']['start_idx']=sub_start
					new_row['subject_entity']['end_idx']=sub_end
					new_row['object_entity']['start_idx']=obj_start
					new_row['object_entity']['end_idx']=obj_end
					augmented_df_per_label.loc[len(augmented_df_per_label)]=new_row
			new_df = pd.concat([new_df, augmented_df_per_label])
		else:
			new_df = pd.concat([new_df, rows_per_label])
	print(new_df['label'].value_counts())

	return new_df.sample(frac=1).reset_index(drop=True)