"""
Created on Wed July 14 10:44:00 2021

@ author : Injo Kim
Data sceintist of Seoultech

Conduct Doc2vec to make node features
"""

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

def read_documents(id) :
    documents_dict = {}
    for id_ in id :
        doc = open(r"readme/" + str(id_) + ".txt", "r", encoding='UTF8').readlines()
        doc = ' '.join([line.replace('\n', ' ') for line in doc])
        documents_dict[id_] = doc

    return documents_dict


def train_doc2vec(text_data) :
    tagged_data = [TaggedDocument(words = word_tokenize(corpus.lower()), tags=[str(text_idx)]) for text_idx, corpus in enumerate(text_data.values())]

    epochs = 100
    embedding_size = 64
    alpha = 0.025
    min_alpha = 0.00025
    distributed_memory = 1

    model = Doc2Vec(size=embedding_size, alpha=alpha, min_alpha=min_alpha, min_count=1, dm=distributed_memory)
    model.build_vocab(tagged_data)

    print('================= Doc2Vec start=================')
    for epoch in range(epochs) :
        model.train(tagged_data, total_examples=model.corpus_count, epochs=model.iter)
        model.alpha -= 0.0002
        
        if epoch % 10 == 0 :
            print('Doc2vec iteration {}'.format(epoch))

    model.save('model/doc2vec_model/readme_encoder_64.model')


def embedding_readme(text_data) :
    #model = 
    model = Doc2Vec.load('model/doc2vec_model/readme_encoder_64.model')
    embedding_vector = [model.infer_vector(word_tokenize(text.lower())) for text in text_data.values()]
    
    return embedding_vector

