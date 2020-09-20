import re
import string
import os
import pandas as pd

class TrainExample:
    def __init__(self, text, keyword, location, target):
        self.text = text
        self.keyword = keyword
        self.location = location
        self.target = target

class TestExample:
    def __init__(self, id, text, keyword, location):
        self.id = id
        self.text = text
        self.keyword = keyword
        self.location = location

class DataExample:

    def clean_data(self, text: str):
        text = text.lower()
        text = re.sub("\[.*?\]", "", text)
        text = re.sub("https?://\S+|www\.\S+", "", text)
        text = re.sub("<.*?>+", "", text)
        text = re.sub('[%s]'.format(re.escape(string.punctuation)), "", text)
        text = re.sub("\n", "", text)
        text = re.sub("\w*\d\w*", "", text)
        return text

    def remove_emoji(self, text):
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        return re.sub(emoji_pattern, "", text)

    def expand_contractions(self, text):
        contractions = {
            "ain't": "am not / are not / is not / has not / have not",
            "aren't": "are not / am not",
            "can't": "can not",
            "'cause": "because",
            "could've": "could have",
            "couldn't": "could not",
            "couldn't've": "could not have",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hadn't've": "had not have",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he had / he would",
            "he'd've": "he would have",
            "he'll": "he shall / he will",
            "he'll've": "he shall have / he will have",
            "he's": "he has / he is",
            "how'd": "how did",
            "how'd'y": "how do you",
            "how'll": "how will",
            "how's": "how has / how is / how does",
            "I'd": "I had / I would",
            "I'd've": "I would have",
            "I'll": "I shall / I will",
            "I'll've": "I shall have / I will have",
            "I'm": "I am",
            "I've": "I have",
            "isn't": "is not",
            "it'd": "it had / it would",
            "it'd've": "it would have",
            "it'll": "it shall / it will",
            "it'll've": "it shall have / it will have",
            "it's": "it has / it is",
            "let's": "let us",
            "ma'am": "madam",
            "mayn't": "may not",
            "might've": "might have",
            "mightn't": "might not",
            "mightn't've": "might not have",
            "must've": "must have",
            "mustn't": "must not",
            "mustn't've": "must not have",
            "needn't": "need not",
            "needn't've": "need not have",
            "o'clock": "of the clock",
            "oughtn't": "ought not",
            "oughtn't've": "ought not have",
            "shan't": "shall not",
            "sha'n't": "shall not",
            "shan't've": "shall not have",
            "she'd": "she had / she would",
            "she'd've": "she would have",
            "she'll": "she shall / she will",
            "she'll've": "she shall have / she will have",
            "she's": "she has / she is",
            "should've": "should have",
            "shouldn't": "should not",
            "shouldn't've": "should not have",
            "so've": "so have",
            "so's": "so as / so is",
            "that'd": "that would / that had",
            "that'd've": "that would have",
            "that's": "that has / that is",
            "there'd": "there had / there would",
            "there'd've": "there would have",
            "there's": "there has / there is",
            "they'd": "they had / they would",
            "they'd've": "they would have",
            "they'll": "they shall / they will",
            "they'll've": "they shall have / they will have",
            "they're": "they are",
            "they've": "they have",
            "to've": "to have",
            "wasn't": "was not",
            "we'd": "we had / we would",
            "we'd've": "we would have",
            "we'll": "we will",
            "we'll've": "we will have",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what'll": "what shall / what will",
            "what'll've": "what shall have / what will have",
            "what're": "what are",
            "what's": "what has / what is",
            "what've": "what have",
            "when's": "when has / when is",
            "when've": "when have",
            "where'd": "where did",
            "where's": "where has / where is",
            "where've": "where have",
            "who'll": "who shall / who will",
            "who'll've": "who shall have / who will have",
            "who's": "who has / who is",
            "who've": "who have",
            "why's": "why has / why is",
            "why've": "why have",
            "will've": "will have",
            "won't": "will not",
            "won't've": "will not have",
            "would've": "would have",
            "wouldn't": "would not",
            "wouldn't've": "would not have",
            "y'all": "you all",
            "y'all'd": "you all would",
            "y'all'd've": "you all would have",
            "y'all're": "you all are",
            "y'all've": "you all have",
            "you'd": "you had / you would",
            "you'd've": "you would have",
            "you'll": "you shall / you will",
            "you'll've": "you shall have / you will have",
            "you're": "you are",
            "you've": "you have"
        }
        contraction_re = re.compile("|".join(list(contractions.keys())))
        def replace(match):
            return contractions[match.group(0)]
        text = re.sub(contraction_re, replace, text)
        return text
        

    def _read_csv(self, data_file):
        lines = []
        data_df = pd.read_csv(data_file).fillna("")
        texts = data_df["text"]
        data_df["text"] = [self.clean_data(text) for text in texts]
        data_df["text"] = [self.remove_emoji(text) for text in texts]
        data_df["text"] = [self.expand_contractions(text) for text in texts]

        for i, line in enumerate(data_df.values):
            line = list(line)
            if line == []:
                continue
            new_line = [line[0]]
            for l in line[1:]:
                if type(l) is str:
                    l = l.replace("#", " ")
                new_line.append(l)
            lines.append(list(new_line))
        return lines



class TrainFeatures(DataExample):
    def __init__(self):
        self.labels = [0, 1]
        self.train_examples = []
    def get_train_example(self, data_dir):
        train_data = self._read_csv(os.path.join(data_dir, "train.csv"))
        for data in train_data:
            if len(data) < 5 or (type(data[4]) is not int):
                print(len(data), data)
                continue

            id, keyword, location, text, target = data[0], data[1], data[2], data[3], int(data[4])
            self.train_examples.append(
                TrainExample(
                    text=text,
                    keyword=keyword,
                    location=location,
                    target=target
                )
            )
    def get_train_features(self, data_dir, max_sequence_len, tokenizer):
        self.get_train_example(data_dir=data_dir)

        train_features = []
        for example in self.train_examples:
            text, keyword, location, target = example.text, example.keyword, example.location, example.target
            text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
            keyword = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(keyword))
            location = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(location))
            len_text = max_sequence_len - len(keyword) - len(location) - 3
            len_padding = 0
            if len_text >= len(text):
                len_padding = len_text - len(text)
            else:
                text = text[:len_text]

            text_append = keyword + location
            input_ids = [101] + text + [102] + text_append + [102]
            token_type_ids = [0] * (len(text) + 2) + [1] * (len(text_append) + 1)
            attention_mask = [1] * len(input_ids)

            if len_padding != 0:
                input_ids = input_ids + [0] * len_padding
                token_type_ids = token_type_ids + [0] * len_padding
                attention_mask = attention_mask + [0] * len_padding

            # target = torch.nn.functional.one_hot(target)
            train_features.append({
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
                "target": target
            })
        return train_features


class TestFeatures(DataExample):
    def __init__(self):
        self.test_examples = []
    def get_test_example(self, data_dir):
        test_data = self._read_csv(os.path.join(data_dir, "test.csv"))
        self.test_examples = []
        for data in test_data:
            if len(data) < 4:
                # id, keyword, location, text = data[0], "", "", data[2]
                print(len(data), data)
                continue
            id, keyword, location, text = data[0], data[1], data[2], data[3]
            self.test_examples.append(
                TestExample(
                    id=id,
                    text=text,
                    keyword=keyword,
                    location=location
                )
            )
    def get_test_features(self, data_dir, max_sequence_len, tokenizer):
        self.get_test_example(data_dir=data_dir)
        print(len(self.test_examples))
        test_features = []
        for example in self.test_examples:
            id, text, keyword, location = example.id, example.text, example.keyword, example.location
            id = int(id)
            text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
            keyword = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(keyword))
            location = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(location))
            len_text = max_sequence_len - len(keyword) - len(location) - 3
            len_padding = 0
            if len_text >= len(text):
                len_padding = len_text - len(text)
            else:
                text = text[:len_text]

            text_append = keyword + location
            input_ids = [101] + text + [102] + text_append + [102]
            token_type_ids = [0] * (len(text) + 2) + [1] * (len(text_append) + 1)
            attention_mask = [1] * len(input_ids)

            if len_padding != 0:
                input_ids = input_ids + [0] * len_padding
                token_type_ids = token_type_ids + [0] * len_padding
                attention_mask = attention_mask + [0] * len_padding


            test_features.append({
                "id": id,
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask
            })
        return test_features

