import data_utils
prompts_path = './data/TIMIT/DOC/PROMPTS.TXT'
dict_path = './data/sentence.pkl'
data_utils.build_sentence_dict(prompts_path, dict_path)