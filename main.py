def learnnames():
    textgen.reset()
    textgen.train_from_file(
        textpath,
        name=modelname,
        word_level=False,
        max_length=50,
        batch_size=1024,
        max_gen_length=25,
        new_model=True,
        rnn_layers=8, #higher == random letters
        rnn_bidirectional=False,
        rnn_size=128, #too high == random letters
        dim_embeddings=250,
        train_size=0.75,
        num_epochs=500,
        gen_epochs=500
    )
    textgen.save(savepath)
def learnbook():
    textgen.reset()
    textgen.train_from_file(
        textpath,
        name=modelname,
        word_level=False,#True for the novels
        max_length=50, #(25) Lower for World Leve
        #max_gen_length=50, #Lower for Word Level, but higher for long texts
        #batch_size=256,
        #max_words=1000,
        new_model=True,
        rnn_layers=4,
        #rnn_bidirectional=True,
        rnn_size=128,
        #dim_embeddings=250,
        num_epochs=10,
        gen_epochs=25,
        train_size=0.75
    )
    textgen.save(savepath)
def printoutput():
    textgen = textgenrnn(
        weights_path=modelname + '_weights.hdf5',
        vocab_path=modelname+'_vocab.json',
        config_path=modelname+'_config.json'
    )
    if "names" in text:
        textgen.generate_to_file(outputpath, max_gen_length=200, n=250, temperature=.5)
        cleantext()
    else:
        textgen.generate_to_file(outputpath, max_gen_length=250, n=5, temperature=.2)

def cleantext():
    cleanpath = './outputs/' + text + '.txt'
    masterfile = open(textpath, "r", encoding="UTF-8")
    uncleanfile = open(outputpath, "r", encoding="UTF-8")
    cleanfile = open(cleanpath, "w+", encoding="UTF-8")

    with uncleanfile as file1:
        with masterfile as file2:
            same = set(file1).difference(file2)
    same.discard('\n')

    words = []
    for line in same:
        words.append(line)
    words.sort()

    for i in words:
        cleanfile.write(i)


    masterfile.close()
    uncleanfile.close()
    cleanfile.close()
    os.remove(outputpath)


from textgenrnn import textgenrnn
import os

textgen = textgenrnn()
text = 'lotr'
learn = True

newpath= r'./models/'+text
if not os.path.exists(newpath):
    os.makedirs(newpath)

modelname = './models/' + text +'/'+ text + '_model'
textpath = './texts/' + text + '.txt'
savepath = './models/' + text +'/'+ text+ '.hdf5'

if "names" in text:
    outputpath = './outputs/' + text + 'Unclean.txt'
else:
    outputpath = './outputs/' + text + '.txt'


if learn == True:
    if "names" in text:
        learnnames()
    else:
        learnbook()
printoutput()