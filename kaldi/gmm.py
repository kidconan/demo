'''@file gmm.py
contains the functionality for Kaldi GMM training, aligning and testing'''

from abc import ABCMeta, abstractproperty
import os

class GMM(object, metaclass=ABCMeta):
    '''an abstract class for a kaldi GMM'''

    def __init__(self, conf):
        '''
        KaldiGMM constructor

        Args:
            conf: the general configurations
        '''

        self.conf = conf

    def train(self):
        '''train the GMM'''

        #save the current dir
        current_dir = os.getcwd()

        #go to kaldi egs dir
        os.chdir(self.conf.get('directories', 'kaldi_egs'))

        #train the GMM
        os.system('%s --cmd %s --config %s/config/%s %s %s %s %s %s' %(
            self.trainscript, 
            self.conf.get('general', 'cmd'), 
            current_dir, self.conf_file, 
            self.trainops,
            self.conf.get('directories', 'train_features')
            + '/' + self.conf.get('gmm-features', 'name'),
            self.conf.get('directories', 'language'),
            self.parent_gmm_alignments,
            self.conf.get('directories', 'expdir') + '/' + self.name))

        #build the decoding graphs
        os.system('utils/mkgraph.sh %s %s %s %s/graph' % (
            self.graphopts, self.conf.get('directories', 'language_test'),
            self.conf.get('directories', 'expdir') + '/' + self.name,
            self.conf.get('directories', 'expdir') + '/' + self.name))

        #go back to working dir
        #print("train finish")
        os.chdir(current_dir)

    def align(self):
        '''use the GMM to align the training utterances'''

        #save the current dir
        current_dir = os.getcwd()

        #go to kaldi egs dir
        os.chdir(self.conf.get('directories', 'kaldi_egs'))

        #do the alignment
        os.system('''%s --nj %s --cmd %s --config %s/config/ali_%s %s %s %s %s %s/ali''' % (
                self.alignscript,
                self.conf.get('general', 'train_num_jobs'),
                self.conf.get('general', 'cmd'), current_dir, self.conf_file,
                self.alignops,
                self.conf.get('directories', 'train_features') + '/'
                + self.conf.get('gmm-features', 'name'),
                self.conf.get('directories', 'language'),
                self.conf.get('directories', 'expdir') + '/' + self.name,
                self.conf.get('directories', 'expdir') + '/' + self.name))

        #convert alignments (transition-ids) to pdf-ids
        for i in range(int(self.conf.get('general', 'train_num_jobs'))):
            os.system('''gunzip -c %s/ali/ali.%d.gz | ali-to-pdf %s/ali/final.mdl ark:- ark,t:- | gzip >  %s/ali/pdf.%d.gz''' % (
                    self.conf.get('directories', 'expdir') + '/' + self.name,
                    i+1, self.conf.get('directories', 'expdir') + '/'
                    + self.name, self.conf.get('directories', 'expdir')
                    + '/' + self.name, i+1))

        #go back to working dir
        #print("align finish")
        os.chdir(current_dir)

    def test(self):
        '''test the GMM on the testing set'''

        #save the current dir
        current_dir = os.getcwd()

        #go to kaldi egs dir
        os.chdir(self.conf.get('directories', 'kaldi_egs'))

        os.system('''%s --cmd %s --nj %s %s %s/graph %s %s/decode | tee %s/decode.log || exit 1;''' % (
            self.decodescript,
            self.conf.get('general', 'cmd'),
            self.conf.get('general', 'decode_num_jobs'),
            self.decodeops,
            self.conf.get('directories', 'expdir') + '/' + self.name,
            self.conf.get('directories', 'test_features') + '/'
            + self.conf.get('gmm-features', 'name'),
            self.conf.get('directories', 'expdir') + '/' + self.name,
            self.conf.get('directories', 'expdir') + '/'  + self.name))

        #go back to working dir
        #print("test finish")
        os.chdir(current_dir)

    @abstractproperty
    def name(self):
        '''the name of the GMM'''
        pass

    @abstractproperty
    def trainscript(self):
        '''the script used for training the GMM'''
        p

    @abstractproperty
    def alignscript(self):
        '''the script used for aligning the GMM'''
        pass
    
    @abstractproperty
    def decodescript(self):
        '''the script used for decoding the GMM'''
        pass

    @abstractproperty
    def conf_file(self):
        '''the configuration file for this GMM'''
        pass

    @abstractproperty
    def parent_gmm_alignments(self):
        '''the path to the parent GMM model (empty for monophone GMM)'''
        pass

    @abstractproperty
    def trainops(self):
        '''the extra options for GMM training'''
        pass

    @abstractproperty
    def alignops(self):
        '''the extra options for GMM training'''
        pass

    @abstractproperty
    def decodeops(self):
        '''the extra options for GMM training'''
        pass

    @abstractproperty
    def graphopts(self):
        '''the extra options for the decoding graph creation'''
        pass

class MonoGmm(GMM):
    ''' a class for the monophone GMM'''

    @property
    def name(self):
        return self.conf.get('mono_gmm', 'name')

    @property
    def trainscript(self):
        return 'steps/train_mono.sh --nj %s' % (self.conf.get('general', 'train_num_jobs'))

    @property
    def alignscript(self):
        return 'steps/align_si.sh'

    @property
    def decodescript(self):
        return 'steps/decode.sh'

    @property
    def conf_file(self):
        return 'mono.conf'

    @property
    def parent_gmm_alignments(self):
        return ''

    @property
    def trainops(self):
        return '--totgauss %s' % ( self.conf.get('mono_gmm', 'tot_gauss') )

    @property
    def alignops(self):
        return ('--boost-silence %s' % ( self.conf.get('mono_gmm', 'boost_silence') ))

    @property
    def decodeops(self):
        return ''

    @property
    def graphopts(self):
        return '--mono'

class TriGmm(GMM):
    '''a class for the triphone GMM'''

    @property
    def name(self):
        return self.conf.get('tri_gmm', 'name')

    @property
    def trainscript(self):
        return 'steps/train_deltas.sh'

    @property
    def alignscript(self):
        return 'steps/align_si.sh'

    @property
    def decodescript(self):
        return 'steps/decode.sh'

    @property
    def conf_file(self):
        return 'tri.conf'

    @property
    def parent_gmm_alignments(self):
        return (self.conf.get('directories', 'expdir') + '/'
                + self.conf.get('mono_gmm', 'name') + '/ali')

    @property
    def trainops(self):
        return (self.conf.get('tri_gmm', 'num_leaves') + ' '
                + self.conf.get('tri_gmm', 'tot_gauss'))

    @property
    def alignops(self):
        return ''

    @property
    def decodeops(self):
        return ''

    @property
    def graphopts(self):
        return ''

class LdaGmm(GMM):
    '''a class for the LDA+MLLT GMM'''

    @property
    def name(self):
        return self.conf.get('lda_mllt', 'name')

    @property
    def trainscript(self):
        return 'steps/train_lda_mllt.sh'

    @property
    def alignscript(self):
        return 'steps/align_si.sh'

    @property
    def decodescript(self):
        return 'steps/decode.sh'

    @property
    def conf_file(self):
        return 'lda_mllt.conf'

    @property
    def parent_gmm_alignments(self):
        return (self.conf.get('directories', 'expdir') + '/'
                + self.conf.get('tri_gmm', 'name') + '/ali')

    @property
    def trainops(self):
        return '%s %s %s'% (
            self.conf.get('lda_mllt', 'context_ops') , 
            self.conf.get('lda_mllt', 'num_leaves') , 
            self.conf.get('lda_mllt', 'tot_gauss'))

    @property
    def alignops(self):
        return ('--use-graphs %s' % ( self.conf.get('lda_mllt','use_graphs') ))

    @property
    def decodeops(self):
        return ''

    @property
    def graphopts(self):
        return ''

class SatGmm(GMM):
    '''a class for the LDA+MLLT+SAT GMM'''

    @property
    def name(self):
        return self.conf.get('lda_mllt_sat', 'name')

    @property
    def trainscript(self):
        return 'steps/train_sat.sh'

    @property
    def alignscript(self):
        return 'steps/align_fmllr.sh'

    @property
    def decodescript(self):
        return 'steps/decode_fmllr.sh'
    @property
    def conf_file(self):
        return 'lda_mllt_sat.conf'

    @property
    def parent_gmm_alignments(self):
        return (self.conf.get('directories', 'expdir') + '/'
                + self.conf.get('lda_mllt', 'name') + '/ali')

    @property
    def trainops(self):
        return (self.conf.get('lda_mllt_sat', 'num_leaves') + ' '
                + self.conf.get('lda_mllt_sat', 'tot_gauss'))

    @property
    def alignops(self):
        return ''

    @property
    def decodeops(self):
        return ''

    @property
    def graphopts(self):
        return ''

class Ubm(GMM):
    ''' a class prepare for SGMM2'''

    @property
    def name(self):
        return self.conf.get('ubm', 'name')

    @property
    def trainscript(self):
        return 'steps/train_ubm.sh'

    @property
    def alignscript(self):
        return 'steps/align_si.sh'

    @property
    def decodescript(self):
        return 'steps/decode.sh'

    @property
    def conf_file(self):
        return 'ubm4.conf'

    @property
    def parent_gmm_alignments(self):
        return (self.conf.get('directories', 'expdir') + '/'
                + self.conf.get('lda_mllt_sat', 'name') + '/ali')

    @property
    def trainops(self):
        return ''

    @property
    def alignops(self):
        return ''

    @property
    def decodeops(self):
        return ''

    @property
    def graphopts(self):
        return ''

class SGmm2(GMM):
    '''a class for the sGMM2'''
    
    def pre_work(self):
        current_dir = os.getcwd()

        #go to kaldi egs dir
        os.chdir(self.conf.get('directories', 'kaldi_egs'))

        #train the GMM
        os.system('%s --nj %s --cmd %s --config %s/config/%s %s %s %s %s %s' %(
            'steps/train_ubm.sh', 
            self.conf.get('general', 'train_num_jobs'),
            self.conf.get('general', 'cmd'), 
            current_dir, 'ubm4.conf', 
            self.conf.get('ubm', 'tot_gauss'),
            self.conf.get('directories', 'train_features')
            + '/' + self.conf.get('gmm-features', 'name'),
            self.conf.get('directories', 'language'),
            self.conf.get('directories', 'expdir') + '/'
                + self.conf.get('lda_mllt_sat', 'name') + '/ali',
            self.conf.get('directories', 'expdir') + '/' + self.conf.get('ubm','name')))
        os.chdir(current_dir)

    @property
    def name(self):
        return self.conf.get('sgmm2', 'name')

    @property
    def trainscript(self):
        return 'steps/train_sgmm2.sh'

    @property
    def alignscript(self):
        return 'steps/align_sgmm2.sh'

    @property
    def decodescript(self):
        return 'steps/decode_sgmm2.sh'
    
    @property
    def conf_file(self):
        return 'sgmm2_4.conf'

    @property
    def parent_gmm_alignments(self):
        #For ubm, alignment isn't used
        return ( (self.conf.get('directories', 'expdir') + '/'+ self.conf.get('lda_mllt_sat', 'name') + '/ali') +' ' + \
        self.conf.get('directories', 'expdir') + '/'
                + self.conf.get('ubm', 'name') + '/final.ubm' )

    @property
    def trainops(self):
        return '%s %s'% ( 
            self.conf.get('sgmm2', 'num_leaves') , 
            self.conf.get('sgmm2', 'tot_gauss'))

    @property
    def alignops(self):
        return '--transform-dir %s --use-graphs %s --use-gselect %s' % (
        self.conf.get('directories', 'expdir') + '/' + self.conf.get('lda_mllt_sat', 'name') + '/ali',
        self.conf.get('sgmm2','use_graphs'),
        self.conf.get('sgmm2','use_gselect')
        )

    @property
    def decodeops(self):
        return '--transform-dir %s' % (
        self.conf.get('directories', 'expdir') + '/' + self.conf.get('lda_mllt_sat', 'name') + '/decode')

    @property
    def graphopts(self):
        return ''
