
# from google.colab import drive
# drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# clone the repo for running evaluation
# examples
# git("status")
# # , "-b", "jb_2.5"
# git("clone", "https://github.com/AI4Bharat/indicTrans.git")
# # %cd indicTrans
# # clone requirements repositories
# git("clone", "https://github.com/anoopkunchukuttan/indic_nlp_library.git")
# git("clone", "https://github.com/anoopkunchukuttan/indic_nlp_resources.git")
# git("clone", "https://github.com/rsennrich/subword-nmt.git")
# %cd ..

# Commented out IPython magic to ensure Python compatibility.
# Install the necessary libraries
# pip install sacremoses pandas mock sacrebleu tensorboardX pyarrow indic-nlp-library
# pip install mosestokenizer subword-nmt
# Install fairseq from source
# git("clone", "https://github.com/pytorch/fairseq.git")
# %cd fairseq
# !git checkout da9eaba12d82b9bfc1442f0e2c6fc1b895f4d35d
# pip install ./
#! pip install xformers
# %cd ..

# add fairseq folder to python path
import os
os.environ['PYTHONPATH'] += "C:/Users/Kanak/Downloads/SIH/fairseq"
# sanity check to see if fairseq is installed
# from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils

# Commented out IPython magic to ensure Python compatibility.
# download the indictrans model


# downloading the indic-en model
#!wget https://storage.googleapis.com/samanantar-public/V0.3/models/indic-en.zip
#!unzip indic-en.zip

# downloading the en-indic model
# !wget https://storage.googleapis.com/samanantar-public/V0.3/models/en-indic.zip
# !unzip en-indic.zip

# # downloading the indic-indic model
# !wget https://storage.googleapis.com/samanantar-public/V0.3/models/m2m.zip
# !unzip m2m.zip

# %cd indicTrans

# pip install triton

from indicTrans.inference.engine import Model

indic2en_model = Model(expdir="C:/Users/Kanak/Downloads/indic-en-20220822T072936Z-002/indic-en")

ta_sents = ['அவனுக்கு நம்மைப் தெரியும் என்று தோன்றுகிறது',
            "நாங்கள் உன்னைத் பின்பற்றுவோம் (அ) தொடர்வோம். ",
            'உங்களுக்கு உங்கள் அருகில் இருக்கும் ஒருவருக்கோ இத்தகைய அறிகுறிகள் தென்பட்டால், வீட்டிலேயே இருப்பது, கொரோனா வைரஸ் தொற்று பிறருக்கு வராமல் தடுக்க உதவும்.']


indic2en_model.batch_translate(ta_sents, 'ta', 'en')

ta_paragraph = """ଏକ ପାରାଗ୍ରାଫ୍ ହେଉଛି ଏକ ନିର୍ଦ୍ଦିଷ୍ଟ ବିନ୍ଦୁ କିମ୍ବା ଧାରଣା ସହିତ ଲିଖିତ ଲେଖାରେ ଏକ ଆତ୍ମ-ଧାରଣ ଏକକ | ଏକ ଅନୁଚ୍ଛେଦ ତିନି କିମ୍ବା ଅଧିକ ବାକ୍ୟକୁ ନେଇ ଗଠିତ | ଯଦିଓ କ language ଣସି ଭାଷାର ବାକ୍ୟବିନ୍ୟାସ ଦ୍ୱାରା ଆବଶ୍ୟକ ହୁଏ ନାହିଁ, ଅନୁଚ୍ଛେଦଗୁଡ଼ିକ ସାଧାରଣତ formal ଆନୁଷ୍ଠାନିକ ଲେଖାର ଏକ ଆଶାକରାଯାଉଥିବା ଅଂଶ, ଲମ୍ବା ଗଦ୍ୟର ଆୟୋଜନ ପାଇଁ ବ୍ୟବହୃତ ହୁଏ |"""

indic2en_model.translate_paragraph(ta_paragraph, 'ta', 'en')

indic2en_model.translate_paragraph('मेरा नाम जोकर 1970 में बनी हिन्दी भाषा की नाट्य फिल्म है।', 'hi', 'en')

# !pip install colabcode
# !pip install fastapi

# from colabcode import ColabCode
# from fastapi import FastAPI

# cc = ColabCode(port=12000, code=False)

# app = FastAPI()

# @app.get("/")
# async def read_root():
#   return {"message": "Subscribe to @1littlecoder"}

# cc.run_app(app=app)

