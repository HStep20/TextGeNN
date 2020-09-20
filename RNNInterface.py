from textgenrnn import textgenrnn
import os
import time
import tensorflow as tf


class RNNInterface:
    def __init__(self, text, learn):
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
        self.text = text
        self.learn = learn
        self.model_name = './models/' + text + '/' + text + '_model'
        self.textpath = './texts/' + text + '.txt'
        self.save_path = './models/' + text + '/' + text + '.hdf5'
        self.output_path = './outputs/' + text + 'Unclean.txt'
        self.new_model_path = r'./models/' + text
        self.tg = textgenrnn()
        self.start_time = time.time()
        self.name_time = 0

        self.rnnLayers = 8 # Number of Recursive Neural Network Layers | Higher == random letters
        self.rnnSize = 128  # Size of RNN Layers | Higher == Random Letters
        self.batchSize = 512  #
        self.trainSize = .7  # Uses x% of data to learn from. 100% means exact same output as input
        self.numEpochs = 100
        self.genEpochs = 50  # spits out output every x epochs
        self.maxInputLength = 50
        self.maxGenLength = 25
        self.bidirectional = True

        # Output Variables
        self.outputMaxGenLength = 200
        self.outputInitialCount = 250
        self.outputTemp = .75

    def generateoutput(self):
        if self.learn:
            self.learnnames()
        self.name_time = time.time()
        textgen = textgenrnn(
            weights_path=self.model_name + '_weights.hdf5',
            vocab_path=self.model_name + '_vocab.json',
            config_path=self.model_name + '_config.json'
        )
        textgen.generate_to_file(self.output_path, max_gen_length=self.outputMaxGenLength, n=self.outputInitialCount,
                                 temperature=self.outputTemp)
        # TODO - Add something to specify starting letters of names with the textgenrnn lib
        self.cleantext()

    def learnnames(self):
        if not os.path.exists(self.new_model_path):
            os.makedirs(self.new_model_path)
        self.tg.reset()
        self.tg.train_from_file(
            self.textpath,
            name=self.model_name,
            word_level=False,
            max_length=self.maxInputLength,
            batch_size=self.batchSize,
            max_gen_length=self.maxGenLength,
            new_model=True,
            rnn_layers=self.rnnLayers,
            rnn_bidirectional=self.bidirectional,
            rnn_size=self.rnnSize,
            dim_embeddings=250,
            train_size=self.trainSize,
            num_epochs=self.numEpochs,
            gen_epochs=self.genEpochs
        )
        self.tg.save(self.save_path)

    def cleantext(self):
        cleanpath = './outputs/' + self.text + ", layers-" + str(self.rnnLayers) + ", size-" + str(
            self.rnnSize) + ", epochs-" + str(self.numEpochs) + ", bidirectional-" + str(
            self.bidirectional) + ", outputTemperature-" + str(self.outputTemp) + '.txt'
        masterfile = open(self.textpath, "r", encoding="UTF-8")
        uncleanfile = open(self.output_path, "r", encoding="UTF-8")
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
            "Time to Learn: " + ("%s2 seconds" % round((time.time() - self.start_time), 2)) +
            "\nTime to Learn: " + ("%s2 minutes" % round(((time.time() - self.start_time) / 60), 2)) +
            "\nTime to Generate Names: " + ("%s1 seconds" % round((time.time() - self.name_time), 2)) +
            "\nTime to Generate Names: " + ("%s1 minutes" % round(((time.time() - self.name_time) / 60), 2)) +
            "\n\n========== Model Info ==========" +
            "\nrnnLayers: " + str(self.rnnLayers) +
            "\nrnnSize: " + str(self.rnnSize) +
            "\nbatchSize: " + str(self.batchSize) +
            "\ntrainSize: " + str(self.trainSize) +
            "\nbidirectional: " + str(self.bidirectional) +
            "\nnumEpochs: " + str(self.numEpochs) +
            # "\nmaxInputLength: " + maxInputLength +
            "\nmaxGenLength: " + str(self.maxGenLength) +
            "\n\n========== Output Info ==========" +
            "\nOutput Temp: " + str(self.outputTemp) +
            # "\nOutput Count: " + str(outputInitialCount) +
            "\nOutput Total: " + str(len(words)) +

            "\n===================================================\n\n"
        )

        for i in words:
            cleanfile.write(i)

        masterfile.close()
        uncleanfile.close()
        cleanfile.close()
        os.remove(self.output_path)
