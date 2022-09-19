# CODIGO DE TREINAMENTO PARA MODELO CRF

import pandas as pd
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import joblib
import math

import dodfSkweak
import re

import string

import spacy


class CRFContratos():
    '''
    CRFContratos um objeto para criacao e aplicacao de um modelo CRF a partir de supervisao fraca

    Atributos: 
        model = representa o modelo CRF gerado ou carregado
    '''

    def __init__(self):
        self.model = None

    def extract_features(self, txt):
        '''
        extrai a features de cada palvara de um texto especifico

        parametros
            txt: um texto em formato de string 

        return: 
            vetor de dicionarios coma s features de cada palavar do txt
        '''
        txt_features = []
        for j in range(len(txt)):
            word_feat = {
                'word': txt[j].lower(),
                'istitle()': txt[j].istitle(),
                'all_capital': txt[j].isupper(),
                'isdigit': txt[j].isdigit(),
                'BOS': True if (j-1 <= 0 or (txt[j-1] == '.' and not(txt[j-2].isdigit()))) else False,
                'EOS': True if (j+1 >= len(txt) or (txt[j+1] == '.' and (j+2 < len(txt) and not(txt[j+2].isdigit())))) else False,
                'ispunctuation': txt[j] in string.punctuation,
                'wordlength': len(txt[j]),
            }
            if j > 0:
                word_feat.update({
                    '-1.word()': txt[j-1].lower(),
                    '-1.istitle()': txt[j-1].istitle(),
                    '-1.all_capital': txt[j-1].isupper(),
                    '-1.isdigit': txt[j-1].isdigit(),
                    '-1.BOS': True if (j-2 <= 0 or (txt[j-2] == '.' and not(txt[j-3].isdigit()))) else False,
                    '-1.EOS': True if (j >= len(txt) or (txt[j] == '.' and (j+1 < len(txt) and not(txt[j+1].isdigit())))) else False,
                    '-1.ispunctuation': txt[j-1] in string.punctuation,
                    '-1.wordlength': len(txt[j-1]),
                })
            if j < (len(txt)-1):
                word_feat.update({
                    '+1.word()': txt[j+1].lower(),
                    '+1.istitle()': txt[j+1].istitle(),
                    '+1.all_capital': txt[j+1].isupper(),
                    '+1.isdigit': txt[j+1].isdigit(),
                    '+1.BOS': True if (j <= 0 or (txt[j] == '.' and not(txt[j-1].isdigit()))) else False,
                    '+1.EOS': True if (j+2 >= len(txt) or (txt[j+2] == '.' and (j+3 < len(txt) and not(txt[j+3].isdigit())))) else False,
                    '+1.ispunctuation': txt[j+1] in string.punctuation,
                    '+1.wordlength': len(txt[j+1]),
                })
            txt_features.append(word_feat)
        return txt_features

    def number_of_digits(self, s):
        """ Returns number of digits in string. """
        return sum(c.isdigit() for c in s)

    def get_base_feat(self, word):
        """ Returns base features of a word. """
        d = {
            'word': word.lower(),
            'is_title': word.istitle(),
            'is_upper': word.isupper(),
            'num_digits': str(self.number_of_digits(word)),
        }
        return d

    def _get_features(cls, sentence):
        """Create features for each word in act.

        Create a list of dict of words features to be used in the predictor module.

        Args:
            act (list): List of words in an act.

        Returns:
            A list with a dictionary of features for each of the words.

        """
        sent_features = []

        for i in range(len(sentence)):
            word = sentence[i]

            word_before = '' if i == 0 else sentence[i-1]
            word_before2 = '' if i <= 1 else sentence[i-2]
            word_before3 = '' if i <= 2 else sentence[i-3]

            word_after = '' if i+1 == len(sentence) else sentence[i+1]
            word_after2 = '' if i+2 >= len(sentence) else sentence[i+2]
            word_after3 = '' if i+3 >= len(sentence) else sentence[i+3]

            word_before = cls.get_base_feat(word_before)
            word_before2 = cls.get_base_feat(word_before2)
            word_before3 = cls.get_base_feat(word_before3)
            word_after = cls.get_base_feat(word_after)
            word_after2 = cls.get_base_feat(word_after2)
            word_after3 = cls.get_base_feat(word_after3)

            word_feat = {
                'bias': 1.0,
                'word': word.lower(),
                'is_title': word.istitle(),
                'is_upper': word.isupper(),
                'is_digit': word.isdigit(),

                'num_digits': str(cls.number_of_digits(word)),
                'has_hyphen': '-' in word,
                'has_dot': '.' in word,
                'has_slash': '/' in word,
            }

            if i > 0:
                word_feat.update({
                    '-1:word': word_before['word'].lower(),
                    '-1:title': word_before['is_title'],
                    '-1:upper': word_before['is_upper'],
                    '-1:num_digits': word_before['num_digits'],
                })
            else:
                word_feat['BOS'] = True

            if i > 1:
                word_feat.update({
                    '-2:word': word_before2['word'].lower(),
                    '-2:title': word_before2['is_title'],
                    '-2:upper': word_before2['is_upper'],
                    '-2:num_digits': word_before2['num_digits'],
                })

            if i > 2:
                word_feat.update({
                    '-3:word': word_before3['word'].lower(),
                    '-3:title': word_before3['is_title'],
                    '-3:upper': word_before3['is_upper'],
                    '-3:num_digits': word_before3['num_digits'],
                })

            if i < len(sentence) - 1:
                word_feat.update({
                    '+1:word': word_after['word'].lower(),
                    '+1:title': word_after['is_title'],
                    '+1:upper': word_after['is_upper'],
                    '+1:num_digits': word_after['num_digits'],
                })
            else:
                word_feat['EOS'] = True

            if i < len(sentence) - 2:
                word_feat.update({
                    '+2:word': word_after2['word'].lower(),
                    '+2:title': word_after2['is_title'],
                    '+2:upper': word_after2['is_upper'],
                    '+2:num_digits': word_after2['num_digits'],
                })

            if i < len(sentence) - 3:
                word_feat.update({
                    '+3:word': word_after3['word'].lower(),
                    '+3:title': word_after3['is_title'],
                    '+3:upper': word_after3['is_upper'],
                    '+3:num_digits': word_after3['num_digits'],
                })

            sent_features.append(word_feat)

        return sent_features

    def separate_cols(self, train_x, train_y):
        '''
        separa as colunas "text" e "labels" do dataframe de modo que possam ser utilizadas no treinamento do modelo

        parametros:
            df: um dataframe especifico gerado na função "get_hmm_dataframe()" que possui o texto e a label IOB

        return:
            x: um gerador com as features extraidas do "text" do dataframe
            y: uma lista com as labels extraidas do "labels"(iob) do dataframe
        '''
        x = (self._get_features(train_x[i].split())
             for i in range(len(train_x)))
        y = [train_y[i].split() for i in range(len(train_y))]
        return x, y

    def preprocess_spacy(self, txt):
        '''
        Aplica um processo de preprocessamento do Spacy 

        parametros:
            txt:um vetor com os textos "crus"

        return:
            vet_txt: um vetor com os textos preprocessados seguindo o tekenizador do Spacy
        '''
        nlp = spacy.load('pt_core_news_sm', disable=["ner", "lemmatizer"])
        docs = list(nlp.pipe(txt))
        vet_txt = []
        for doc in docs:
            aux = ""
            for token in doc:
                aux += token.text + ' '
            vet_txt.append(aux)

        return vet_txt

    def get_dataframe(self, dados):
        '''
        aplica a supervisao fraca na base de dados

        parametros:
            dados: um vetor de strings, cada uma representadno um contrato extraido

        return:
            um dataframe com as entidades extraidas, o texto e o iob de cada contrato
        '''
        skw = dodfSkweak.SkweakContratos(dados)

        skw.apply_label_functions()
        skw.train_HMM_Dodf()

        df = skw.get_hmm_dataframe()

        return df['text'], df['labes'], df

    def init_model_lbfgs(self):
        '''
        inicializa o modelo CRF utilizando o algoritimo lbfgs(Gradient descent using the L-BFGS method)
        '''
        self.model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )

    def init_model_l2sgd(self):
        '''
        inicializa o modelo CRF utilizando o algoritimo l2sgd(Stochastic Gradient Descent with L2 regularization term)
        '''
        self.model = sklearn_crfsuite.CRF(
            algorithm='l2sgd',
            c2=1,
            max_iterations=100,
            all_possible_transitions=True,
            verbose=False
        )

    def init_model_from_path(self, path):
        '''
        inicializa o modelo CRF a partir de um modelo ja treina previamente e armazenado na maquina

        parametros:
            path: uma string representado o caminho onde o modelo esta armazenado

        '''
        self.model = joblib.load(str(path), 'r')

    def train_model(self, train_x, train_y):
        '''
        treina o modelo crf a partir de um dataframe

        parametros:
            df: dataframe extraido da base de dados de contratos
        '''
        txt, lbl = self.separate_cols(train_x, train_y)
        self.model.fit(txt, lbl)

    def model_predict(self, txt):
        '''
        realiza uma previsão de uma base de dados utilizando o modelo ja treinado

        parametros:
            vet_txt: vetor de strings, cada uma representadno um contrato 

        return:
            um vetor de previsoes em IOB, com uma previsao para cada contrato do parametro passado
        '''

        x = (self._get_features(txt[i].split())for i in range(len(txt)))

        return self.model.predict(x)

    def test_model(self, train_x, train_y):
        '''
        realiza uma teste de acuracia do modelo ja treinado, a partir de uma base de dados utilizando 
        mostra o f1  score do modelo assim como o resultado de classificacao de cada entidade

        parametros:
            dados: vetor de strings, cada uma representadno um contrato 

        '''
        txt, lbl = self.separate_cols(train_x, train_y)
        lbl_ans = self.model.predict(txt)

        labels = list(self.model.classes_)
        labels.remove('O')

        f1 = metrics.flat_f1_score(
            lbl, lbl_ans, average='weighted', labels=labels)

        print("Model Score:", f1,
              "\n     =========//=========//=========//=========     ")

        print(metrics.flat_classification_report(
            lbl, lbl_ans, labels=labels, digits=3
        ))

    def save_model(self, name):
        '''
        salva o modelo gerado e treinado em formato .pkl

        parametros:
            name: string representado o nome com o qual deseja salvar o modelo

        '''
        nome = str(name)+".pkl"
        joblib.dump(self.model, nome)
