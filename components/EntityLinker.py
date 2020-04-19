#HOW-TO: 
#from EntityLinker import EntityLinker (import class)
#EntityLinker = EntityLinker() (instantiate class)
#EntityLinker(question), where question is the whole question string, would return a tuple(tokenized list of the question with entity replaced, entity)
#!!! Remember that you have to escape the 's present in the string (use "..." or '''...''')

import pandas as pd

class EntityLinker:
	def __init__(self, path_KB, path_QA):
		# qa_file_path = "data/PQ-2H.txt"
		# kb_file_path = "data/2H-kb.txt"
		try:
			self.df_qa = pd.read_csv(path_QA, sep='\t', header=None, names=['question_sentence', 'answer_set', 'answer_path'])
			self.df_qa['answer'] = self.df_qa['answer_set'].apply(lambda x: x.split('(')[0])
			self.df_qa['q_split'] = self.df_qa['question_sentence'].apply(lambda x: x.lower().split(' '))
			self.df_kb = pd.read_csv(path_KB,sep='\s', header=None, names= ['e_subject','relation','e_object'])
			self.create_entity_set()
		except Exception as e:
			print('File path wrong')

	def create_entity_set(self):
		subject_set = set(self.df_kb['e_subject'].unique())
		object_set = set(self.df_kb['e_object'].unique())
		self.entity_set = subject_set.union(object_set)

#Use on the question string and return a list of words from the question (replaced)
	def find_entity(self, question):
		modified_question_list = []
		entity_list = []
		entity = ''
		for idx, item in enumerate(question.split(' ')):
			if item in self.entity_set:
				entity_list.append(item)
		entity = max(entity_list, key=len)
		for item in question.split(' '):
			if item == entity:
				modified_question_list.append('<e>')
			else:
				modified_question_list.append(item)
		return (modified_question_list, entity)
