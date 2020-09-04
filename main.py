from textgenrnn import textgenrnn
import os
import time

start_time = time.time()

rnnLayers = 8 #Number of Recursive Neural Network Layers | Higher == random letters
#Size of RNN Layers | Higher == Random Letters
rnnSize = 256
batchSize = 256 #
trainSize = .7 #Uses x% of data to learn from. 100% means exact same output as input
numEpochs = 100
genEpochs = 10 #spits out output every x epochs
maxInputLength = 50
maxGenLength = 25
bidirectional = True

#Output Variables
outputMaxGenLength = 200
outputInitialCount = 250
outputTemp = .75

def learnnames():
    textgen.reset()
    textgen.train_from_file(
        textpath,
        name=modelname,
        word_level=False,
        max_length=maxInputLength,
        batch_size=batchSize,
        max_gen_length=maxGenLength,
        new_model=True,
        rnn_layers=rnnLayers,
        rnn_bidirectional=bidirectional,
        rnn_size=rnnSize,
        dim_embeddings=250,
        train_size=trainSize,
        num_epochs=numEpochs,
        gen_epochs=genEpochs
    )
    textgen.save(savepath)
def printoutput():
    textgen = textgenrnn(
        weights_path=modelname + '_weights.hdf5',
        vocab_path=modelname+'_vocab.json',
        config_path=modelname+'_config.json'
    )
    textgen.generate_to_file(outputpath, max_gen_length=outputMaxGenLength, n=outputInitialCount, temperature=outputTemp)
    #TODO - Add something to specify starting letters of names with the textgenrnn lib
    cleantext()


def cleantext():
    cleanpath = './outputs/' + text + ", layers-"+str(rnnLayers) + ", size-" + str(rnnSize) + ", epochs-"+str(numEpochs)+ ", bidirectional-"+str(bidirectional) + ", outputTemperature-" + str(outputTemp) + '.txt'
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
    cleanfile.write(
        "Time to execute: " + ("%s2 seconds" % round((time.time() - start_time), 2)) +
        "\nTime to execute: " + ("%s2 minutes" % round(((time.time() - start_time)/60), 2)) +
        "\n\n========== Model Info ==========" +
        "\nrnnLayers: " + str(rnnLayers) +
        "\nrnnSize: " + str(rnnSize) +
        "\nbatchSize: " + str(batchSize) +
        "\ntrainSize: " + str(trainSize) +
        "\nbidirectional: " + str(bidirectional) +
        "\nnumEpochs: " + str(numEpochs) +
        #"\nmaxInputLength: " + maxInputLength +
        "\nmaxGenLength: " + str(maxGenLength) +
        "\n\n========== Output Info =========="+
        "\nOutput Temp: " + str(outputTemp) +
        #"\nOutput Count: " + str(outputInitialCount) +
        "\nOutput Total: " + str(len(words)) +

        "\n===================================================\n\n"
    )


    for i in words:
        cleanfile.write(i)


    masterfile.close()
    uncleanfile.close()
    cleanfile.close()
    os.remove(outputpath)

textgen = textgenrnn()
text = 'slanames'
learn = True

newpath= r'./models/'+text
if not os.path.exists(newpath):
    os.makedirs(newpath)

modelname = './models/' + text +'/'+ text + '_model'
textpath = './texts/' + text + '.txt'
savepath = './models/' + text +'/'+ text+ '.hdf5'
outputpath = './outputs/' + text + 'Unclean.txt'


if learn == True:
    learnnames()

printoutput()