from fastai.text import *

'''
text can't be transformed directly into numbers, so it needs
to be preprocessed
    - the raw text is changed into a list of words (tokenization)
    - the tokens are transformed into numbers (numericalization)
    - the numbers are passed to embedding layers that convert them
      into arrays of floates to be passed to a model

fastai is more focused on using pretrained models
    - it has representations of words, sentences and documents
    - structured on three steps
        - preprocessing data
        - create lm with pretrained weights
        - create other models on top (like classifiers)

raw datasets can be made into a dataset when it is
    - organized into folders in an imagenet style
    - or into csv files with a label and text column

fine-tuning language model
    - we can create a learner object that will directly create
      a model, download the pretrained weights and be ready
      for fine-tuning
'''

# get reviews from imdb
path = untar_data(URLs.IMDB_SAMPLE)
print(path)

# look at imbd IMDB_SAMPLE
df = pd.read_csv(path/'texts.csv')
print(df.head())

# get databunch for lm (only first time)
# data_lm = TextLMDataBunch.from_csv(path, 'texts.csv')
#
# get databunch for classifier
# data_clas = TextClasDataBunch.from_csv(path, 'texts.csv',
#     vocab=data_lm.train_ds.vocab, bs=32)

# does all necessary preprocessing, passes the vocabulary
# (mapping from ids to words) that we want to use (this is
# to ensure that data_clas uses the same dictionary as data_lm)

# save the result (because it's time consuming)
# data_lm.save('data_lm_export.pkl')
# data_clas.save('data_clas_export.pkl')

# reload the results
data_lm = load_data(path, 'data_lm_export.pkl')
data_clas = load_data(path, 'data_clas_export.pkl')

# create learner that creates the model and downloads
# pre-trained weights and be ready for fine-tuning
# learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)
# learn.fit_one_cycle(1, 1e-2)

# unfreeze model and fine-tune it
# learn.unfreeze()
# learn.fit_one_cycle(1, 1e-3)

# evaluate language model with Learner.predict and
# specify amount of words to be guessed
# print(learn.predict("This is a review about", n_words=10))

# save encoder
# learn.save_encoder('ft_enc')

# build learner with data_clas and fine-tuned encoder
learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
learn.load_encoder('ft_enc')

# train
learn.fit_one_cycle(1, 1e-2)
learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(5e-3/2., 5e-3))

# predict on raw text
learn.predict("This was a great movie!")

# save classifier
learn.save_encoder('clas')
