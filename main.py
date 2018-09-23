'''@file main.py
run this file to go through the neural net training procedure, look at the config files in the config directory to modify the settings'''

import os
from six.moves import configparser
from neuralNetworks import nnet
from processing import ark, prepare_data, feature_reader, batchdispenser, target_coder
from shutil import copyfile
from kaldi import gmm

#here you can set which steps should be executed. If a step has been executed in the past the result have been saved and the step does not have to be executed again (if nothing has changed)

#GMMTRAINFEATURES = True 	#required
GMMTRAINFEATURES = True     #required

#GMMTESTFEATURES = False	 	#required if the performance of a GMM is tested
GMMTESTFEATURES = True     #required if the performance of a GMM is tested

#DNNTRAINFEATURES = True 	#required
DNNTRAINFEATURES = False     #required

#DNNTESTFEATURES = True	 	#required if the performance of the DNN is tested
DNNTESTFEATURES = False      #required if the performance of the DNN is tested

#TRAIN_MONO = True 			#required
TRAIN_MONO = True           #required

#ALIGN_MONO = True			#required
ALIGN_MONO = True           #required

#TEST_MONO = False 			#required if the performance of the monphone GMM is tested
TEST_MONO = True           #required if the performance of the monphone GMM is tested

#TRAIN_TRI = True			#required if the triphone or LDA GMM is used for alignments
TRAIN_TRI = True            #required if the triphone or LDA GMM is used for alignments

#ALIGN_TRI = True			#required if the triphone or LDA GMM is used for alignments
ALIGN_TRI = True            #required if the triphone or LDA GMM is used for alignments

#TEST_TRI = False			#required if the performance of the triphone GMM is tested
TEST_TRI = True            #required if the performance of the triphone GMM is tested

#TRAIN_LDA = True			#required if the LDA GMM is used for alignments
TRAIN_LDA = True            #required if the LDA GMM is used for alignments

#ALIGN_LDA = True			#required if the LDA GMM is used for alignments
ALIGN_LDA = True            #required if the LDA GMM is used for alignments

#TEST_LDA = False			#required if the performance of the LDA GMM is tested
TEST_LDA = True            #required if the performance of the LDA GMM is tested

TRAIN_SAT = True

ALIGN_SAT = True

TEST_SAT = True

TRAIN_SGMM2 = True

ALIGN_SGMM2 = True

TEST_SGMM2 = True

#TRAIN_NNET = True			#required
TRAIN_NNET = True           #required

#TEST_NNET = True			#required if the performance of the DNN is tested
TEST_NNET = True            #required if the performance of the DNN is tested

use_dnn= True

#read config file
config = configparser.ConfigParser()
config.read('config/config_AURORA4.cfg')
current_dir = os.getcwd()


#compute the features of the training set for GMM training
if GMMTRAINFEATURES:
    feat_cfg = dict(config.items('gmm-features'))

    print('------- computing GMM training features ----------')
    prepare_data.prepare_data(config.get('directories', 'train_data'), config.get('directories', 'train_features') + '/' + feat_cfg['name'], feat_cfg, feat_cfg['type'], feat_cfg['dynamic'])

    print('------- computing cmvn stats ----------')
    prepare_data.compute_cmvn(config.get('directories', 'train_features') + '/' + feat_cfg['name'])

#compute the features of the training set for DNN training if they are different then the GMM features
if DNNTRAINFEATURES:
    if config.get('dnn-features', 'name') != config.get('gmm-features', 'name'):
        feat_cfg = dict(config.items('dnn-features'))

        print('------- computing DNN training features ----------')
        prepare_data.prepare_data(config.get('directories', 'train_data'), config.get('directories', 'train_features') + '/' + feat_cfg['name'], feat_cfg, feat_cfg['type'], feat_cfg['dynamic'])

        print('------- computing cmvn stats ----------')
        prepare_data.compute_cmvn(config.get('directories', 'train_features') + '/' + feat_cfg['name'])


#compute the features of the training set for GMM testing
if GMMTESTFEATURES:
    feat_cfg = dict(config.items('gmm-features'))

    print('------- computing GMM testing features ----------')
    prepare_data.prepare_data(config.get('directories', 'test_data'), config.get('directories', 'test_features') + '/' + feat_cfg['name'], feat_cfg, feat_cfg['type'], feat_cfg['dynamic'])

    print('------- computing cmvn stats ----------')
    prepare_data.compute_cmvn(config.get('directories', 'test_features') + '/' + feat_cfg['name'])

#compute the features of the training set for DNN testing if they are different then the GMM features
if DNNTESTFEATURES:
    if config.get('dnn-features', 'name') != config.get('gmm-features', 'name'):
        feat_cfg = dict(config.items('dnn-features'))

        print('------- computing DNN testing features ----------')
        prepare_data.prepare_data(config.get('directories', 'test_data'), config.get('directories', 'test_features') + '/' + feat_cfg['name'], feat_cfg, feat_cfg['type'], feat_cfg['dynamic'])

        print('------- computing cmvn stats ----------')
        prepare_data.compute_cmvn(config.get('directories', 'test_features') + '/' + feat_cfg['name'])


# #use kaldi to train the monophone GMM
# mono_gmm = gmm.MonoGmm(config)
# if TRAIN_MONO:
#     print('------- mono gmm train ----------')
#     mono_gmm.train()
#     print('mono train finish')
#     print()

# #use kaldi to test the monophone gmm
# if TEST_MONO:
#     print('------- mono gmm test ----------')
#     mono_gmm.test()
#     print('mono test finish')
#     print()

# #get alignments with the monophone GMM
# if ALIGN_MONO:
#     print('------- mono gmm align ----------')
#     mono_gmm.align()
#     print('mono align finish')
#     print()


# #use kaldi to train the triphone GMM
# tri_gmm = gmm.TriGmm(config)
# if TRAIN_TRI:
#     print('------- tri gmm train ----------')
#     tri_gmm.train()
#     print('tri train finish')
#     print()

# #use kaldi to test the triphone gmm
# if TEST_TRI:
#     print('------- tri gmm test ----------')
#     tri_gmm.test()
#     print('tri test finish')
#     print()

# #get alignments with the triphone GMM
# if ALIGN_TRI:
#     print('------- tri gmm align ----------')
#     tri_gmm.align()
#     print('tri align finish')
#     print()

# #use kaldi to train the LDA+MLLT GMM
# lda_gmm = gmm.LdaGmm(config)
# if TRAIN_LDA:
#     print('------- lda gmm train ----------')
#     lda_gmm.train()
#     print('LDA train finish')
#     print()

# #use kaldi to test the LDA+MLLT gmm
# if TEST_LDA:
#     print('------- lda gmm test ----------')
#     lda_gmm.test()
#     print('LDA test finish')
#     print()

# #get alignments with the LDA+MLLT GMM
# if ALIGN_LDA:
#     print('------- lda gmm align ----------')
#     lda_gmm.align()
#     print('LDA align finish')
#     print()

# sat_gmm = gmm.SatGmm(config)
# if TRAIN_SAT:
#     print('------- sat gmm train ----------')
#     sat_gmm.train()
#     print('SAT train finish')
#     print()

# #use kaldi to test the LDA+MLLT+SAT gmm
# if TEST_SAT:
#     print('------- sat gmm test ----------')
#     sat_gmm.test()
#     print('SAT test finish')
#     print()

# #get alignments with the LDA+MLLT+SAT GMM
# if ALIGN_SAT:
#     print('------- sat gmm align ----------')
#     sat_gmm.align()
#     print('SAT align finish')
#     print()

# sgmm2 = gmm.SGmm2(config)
# if TRAIN_SGMM2:
#     print('------- sgmm2 train ----------')
#     sgmm2.pre_work()
#     sgmm2.train()
#     print('LDA train finish')
#     print()

# #use kaldi to test the LDA+MLLT gmm
# if TEST_SGMM2:
#     print('------- sgmm2 gmm test ----------')
#     sgmm2.test()
#     print('SGMM2 test finish')
#     print()

# #get alignments with the LDA+MLLT GMM
# if ALIGN_SGMM2:
#     print('------- sgmm2 align ----------')
#     sgmm2.align()
#     print('SGMM2 align finish')
#     print()

if use_dnn:
    #get the feature input dim
    reader = ark.ArkReader(config.get('directories', 'train_features') + '/' + config.get('dnn-features', 'name') + '/feats.scp')
    _, features, _ = reader.read_next_utt()
    input_dim = features.shape[1]

    #get number of output labels

    numpdfs = open(config.get('directories', 'expdir') + '/' + config.get('nnet', 'gmm_name') + '/graph/num_pdfs')
    num_labels = numpdfs.read()
    num_labels = int(num_labels[0:len(num_labels)-1])
    numpdfs.close()

    #create the neural net
    print()
    nnet = nnet.Nnet(config, input_dim, num_labels)
    print()

if TRAIN_NNET:

    #only shuffle if we start with initialisation
    if config.get('nnet', 'starting_step') == '0':
        #shuffle the examples on disk
        print('------- shuffling examples ----------')
        prepare_data.shuffle_examples(config.get('directories', 'train_features') + '/' +  config.get('dnn-features', 'name'))

    #put all the alignments in one file
    alifiles = [config.get('directories', 'expdir') + '/' + config.get('nnet', 'gmm_name') + '/ali/pdf.' + str(i+1) + '.gz' for i in range(int(config.get('general', 'train_num_jobs')))]
    alifile = config.get('directories', 'expdir') + '/' + config.get('nnet', 'gmm_name') + '/ali/pdf.all'
    os.system('cat %s > %s' % (' '.join(alifiles), alifile))

    #create a feature reader
    featdir = config.get('directories', 'train_features') + '/' +  config.get('dnn-features', 'name')
    with open(featdir + '/maxlength', 'r') as fid:
        max_input_length = int(fid.read())
    featreader = feature_reader.FeatureReader(featdir + '/feats_shuffled.scp', featdir + '/cmvn.scp', featdir + '/utt2spk', int(config.get('nnet', 'context_width')), max_input_length)

    #create a target coder
    coder = target_coder.AlignmentCoder(lambda x, y: x, num_labels)

    dispenser = batchdispenser.AlignmentBatchDispenser(featreader, coder, int(config.get('nnet', 'batch_size')), alifile)

    #train the neural net
    print('------- training neural net ----------')
    nnet.train(dispenser)
    print('dnn train finish')
    print()


if TEST_NNET:

    #use the neural net to calculate posteriors for the testing set
    print('------- computing state pseudo-likelihoods ----------')
    savedir = config.get('directories', 'expdir') + '/' + config.get('nnet', 'name')
    decodedir = savedir + '/decode'
    if not os.path.isdir(decodedir):
        os.mkdir(decodedir)

    featdir = config.get('directories', 'test_features') + '/' +  config.get('dnn-features', 'name')

    #create a feature reader
    with open(featdir + '/maxlength', 'r') as fid:
        max_length = int(fid.read())
    featreader = feature_reader.FeatureReader(featdir + '/feats.scp', featdir + '/cmvn.scp', featdir + '/utt2spk', int(config.get('nnet', 'context_width')), max_length)

    #create an ark writer for the likelihoods
    if os.path.isfile(decodedir + '/likelihoods.ark'):
        os.remove(decodedir + '/likelihoods.ark')
    writer = ark.ArkWriter(decodedir + '/feats.scp', decodedir + '/likelihoods.ark')

    #decode with te neural net
    nnet.decode(featreader, writer)

    print('------- decoding testing sets ----------')
    #copy the gmm model and some files to speaker mapping to the decoding dir

    os.system('cp %s %s' %(config.get('directories', 'expdir') + '/' + config.get('nnet', 'gmm_name') + '/final.mdl', decodedir))
    os.system('cp -r %s %s' %(config.get('directories', 'expdir') + '/' + config.get('nnet', 'gmm_name') + '/graph', decodedir))
    # os.system('cp %s %s' %(config.get('directories', 'test_features') + '/' +  config.get('dnn-features', 'name') + '/utt2spk', decodedir))
    # os.system('cp %s %s' %(config.get('directories', 'test_features') + '/' +  config.get('dnn-features', 'name') + '/text', decodedir))

    # current_dir = os.getcwd()
    os.chdir(config.get('directories', 'test_features') + '/' +  config.get('dnn-features', 'name'))
    datadir=os.getcwd()
    file_list=os.listdir()
    # print(file_list)
    file_need = list(filter(os.path.isfile,file_list))
    file_need.remove('feats.scp')
    file_need.remove('cmvn.scp')
    for file in file_need:
        copyfile(datadir + '/' + file, decodedir + '/' +file)
    os.chdir(current_dir)


    #change directory to kaldi egs
    os.chdir(config.get('directories', 'kaldi_egs'))

    #decode using kaldi
    os.system('%s/kaldi/decode.sh --cmd %s --nj %s %s/graph %s %s/kaldi_decode | tee %s/decode.log || exit 1;' % (current_dir, config.get('general', 'cmd'), config.get('general', 'decode_num_jobs'), decodedir, decodedir, decodedir, decodedir))

    #get results
    os.system('for x in %s/kaldi_decode*; do [ -d $x ] && echo $x | grep "${1:-.*}" >/dev/null && grep Sum $x/score_*/*.sys 2>/dev/null | utils/best_wer.sh; done' % decodedir)

    #go back to working dir
    os.chdir(current_dir)
    print('dnn test finish')
    print()
