#import py_vncorenlp
import string
import time
import numpy as np
import os

from keras.models import Model, load_model
from keras.layers import Input
from transformers import PhobertTokenizer
from vncorenlp import VnCoreNLP

import constants
from dotenv import load_dotenv

load_dotenv()

CURRENT_DIR = os.getcwd()

class Inference_model:
    def __init__(self, model_name):
        self.load_annotator()
        self.load_tokenizer()
        self.load_inference_model(model_name)
        self.denfine_inference_encoder()
        self.define_inference_decoder()
        
    def load_annotator(self):
        #self.annotator = VnCoreNLP("./resources/vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 
        vncorenlp_svc_host = os.getenv('vncorenlp_svc_host')
        if not vncorenlp_svc_host:
            vncorenlp_svc_host = "http://127.0.0.1"
            
        vncorenlp_svc_port = os.getenv('vncorenlp_svc_port')
        if not vncorenlp_svc_port:
            vncorenlp_svc_port = "8000"
        
        self.__annotator = VnCoreNLP(address=vncorenlp_svc_host, port=int(vncorenlp_svc_port))


    def load_tokenizer(self):
        self.__tokenizer = PhobertTokenizer.from_pretrained('vinai/phobert-large', model_max_length = constants.MODEL_MAX_LENGTH)

    def load_inference_model(self, model_name):
        model = load_model('./resources/models/' + model_name)

        # Get layers
        self.encoder_input_ids = model.input[0]
        self.encoder_emb_layer = model.get_layer('embedding')
        self.encoder_blstm_layer = model.get_layer('bidirectional')
        self.decoder_input_ids = model.input[1]
        self.decoder_emb_layer = model.get_layer('embedding_1')
        self.decoder_lstm_layer = model.get_layer('lstm_1')
        self.decoder_dense_layer = model.get_layer('time_distributed')

    def denfine_inference_encoder(self):
        # Define Inference Encoder
        encoder_emb = self.encoder_emb_layer(self.encoder_input_ids)
        encoder_outputs, enc_forward_h, enc_forward_c, enc_backward_h, enc_backward_c = self.encoder_blstm_layer(encoder_emb)
        self.encoder_inf_model = Model([self.encoder_input_ids] ,[encoder_outputs, enc_forward_h, enc_forward_c, enc_backward_h, enc_backward_c])

    def define_inference_decoder(self):
        # Define Inference Decoder
        decoder_input_forward_h = Input(shape=(constants.LATENT_DIM,))
        decoder_input_forward_c = Input(shape=(constants.LATENT_DIM,))

        decoder_input_ids = Input(shape=(1,), dtype = 'int32')

        decoder_states_inputs = [decoder_input_forward_h, decoder_input_forward_c]

        decoder_emb = self.decoder_emb_layer(decoder_input_ids)
        decoder_outputs, dec_forward_h, dec_forward_c = self.decoder_lstm_layer(decoder_emb, initial_state=decoder_states_inputs)
        decoder_states = [dec_forward_h, dec_forward_c]
        decoder_outputs = self.decoder_dense_layer(decoder_outputs)

        self.decoder_inf_model = Model(
            [decoder_input_ids] + decoder_states_inputs, 
            [decoder_outputs] + decoder_states
        )
    
    def preprocessing(self, text):
        exclude = set(string.punctuation)
        text = ' '.join(text.split()) # remove extra white space
        text =  ''.join(ch for ch in text if ch not in exclude) # remove punctuation
        text = text.lower() # lower text
        text = ' '.join([' '.join(sentence) for sentence in self.__annotator.tokenize(text)]) # word segmentation
        return text

    def tokenize_text(self, text):
        tokens = self.__tokenizer(text, return_attention_mask=False, padding = 'max_length', return_token_type_ids=False, return_tensors = 'np')
        input_ids = tokens['input_ids']
        return input_ids

    def decode_sequence(self, input_ids):
        # Encode the input as state vectors.
        start_time = time.time()
        _, enc_forward_h, enc_forward_c, _, _ = self.encoder_inf_model.predict(input_ids)
        # Generate empty target sequence of length 1
        target_input_id = np.zeros((1,1), dtype = 'int32')
        
        #print(encoder_outputs)
        # Populate the first word of target sequence with the start word.
        target_input_id[0, 0] = constants.START_TOKEN

        stop_condition = False
        decoded_sentence = ''

        while not stop_condition:
            (output_tokens, dec_forward_h, dec_forward_c) = self.decoder_inf_model.predict([target_input_id] + [enc_forward_h, enc_forward_c])

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token =  self.__tokenizer.decode([sampled_token_index])
            #print(f"sampled_token_index = {sampled_token_index}; sampled_token = '{sampled_token}'")
                
            if sampled_token != '</s>' and sampled_token != '<pad>':
                if '@@' in sampled_token:
                    sampled_token = sampled_token.replace('@@', '')
                    decoded_sentence +=  sampled_token
                else:
                    decoded_sentence +=  sampled_token + ' '

            # Exit condition: either hit max length or find the stop word.
            if sampled_token == '</s>' or sampled_token =='<pad>' or len(decoded_sentence.split()) >= constants.MODEL_MAX_LENGTH - 1 :
                stop_condition = True

            # Update the target sequence (of length 1)
            target_input_id = np.zeros((1, 1))
            target_input_id[0, 0] = sampled_token_index

            # Update internal states
            (enc_forward_h, enc_forward_c) = (dec_forward_h, dec_forward_c)
            
        return decoded_sentence[:-1], time.time() - start_time

    def generate_answer(self, question):
        cleaned_quesition = self.preprocessing(question)
        input_ids = self.tokenize_text(cleaned_quesition)
        predicted_sentence, execution_time = self.decode_sequence(input_ids)
        return predicted_sentence, execution_time