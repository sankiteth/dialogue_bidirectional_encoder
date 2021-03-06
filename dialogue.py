#!/usr/bin/env python

import math
import sys
import os
import time

import numpy as np
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.layers import safe_embedding_lookup_sparse as embedding_lookup_unique
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell, MultiRNNCell, DropoutWrapper, BasicLSTMCell
from tensorflow.python.platform import gfile

import data_utils


tf.app.flags.DEFINE_float("learning_rate"             , 0.001 , "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99  , "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm"         , 5.0   , "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size"              , 64    , "Batch size to use during training.")
tf.app.flags.DEFINE_integer("encoder_hidden_units"    , 1024  , "Size of each model layer.")
tf.app.flags.DEFINE_integer("embedding_size"          , 100   , "Number of dimensions in embedding space.")
tf.app.flags.DEFINE_integer("keep_prob"               , 0.5   , "input_keep_prob for a single RNN cell.")
tf.app.flags.DEFINE_integer("num_layers"              , 4     , "Number of layers in the model.")
tf.app.flags.DEFINE_integer("vocab_size"              , 5000  , "English vocabulary size.")
tf.app.flags.DEFINE_integer("num_epochs"              , 20    , "Number of epochs to run")
tf.app.flags.DEFINE_integer("max_inf_target_len"      , 100   , "Max length of targets in the inference")
tf.app.flags.DEFINE_integer("output_pred"             , 1     , "write predicted outputs to a text file")

tf.app.flags.DEFINE_string("data_path" , "data/Training_Shuffled_Dataset.txt", "Data directory")#done
tf.app.flags.DEFINE_string("vocab_path", "data/Vocab_file.txt", "Data directory")#done
tf.app.flags.DEFINE_string("dev_data"  , "data/Validation_Shuffled_Dataset.txt", "Data directory")#done
tf.app.flags.DEFINE_string("train_dir" , "train_dir/", "Training directory.")#done
tf.app.flags.DEFINE_string("embedding_file" , "./embeddings/wordembeddings-dim100.word2vec", "Embedding File")#todo


FLAGS = tf.app.flags.FLAGS

_buckets = [(5, 10), (10, 15), (20, 25), (40, 50), (100, 100)]

class Seq2SeqModel():

	def __init__(self, encoder_cell, decoder_cell, vocab_size, embedding_size, learning_rate, max_gradient_norm,
				num_samples=512,
				bidirectional=True,
				attention=False,
				pretrain=False):

		self.bidirectional = bidirectional
		self.attention = attention

		self.vocab_size = vocab_size
		self.embedding_size = embedding_size
		self.pretrain = pretrain


		self.learning_rate = learning_rate
		self.max_gradient_norm = max_gradient_norm

		self.encoder_cell = encoder_cell
		self.decoder_cell = decoder_cell

		self.loss_fn = None

		self._make_graph()

		self.saver = tf.train.Saver(max_to_keep=10000)

	@property
	def decoder_hidden_units(self):
		return self.decoder_cell.output_size

	def _make_graph(self):

		self._init_placeholders()

		self._init_decoder_tensors_padding_masking()
		self._init_embeddings()

		if self.bidirectional:
			self._init_bidirectional_encoder()
		else:
			self._init_simple_encoder()

		self._init_decoder()

		self._init_optimizer()

	def _init_placeholders(self):
		""" Everything is time-major """
		self.encoder_inputs = tf.placeholder(
			shape=(None, None),
			dtype=tf.int32,
			name='encoder_inputs',
		)
		self.encoder_inputs_length = tf.placeholder(
			shape=(None,),
			dtype=tf.int32,
			name='encoder_inputs_length',
		)

		# required for training, not required for testing
		self.decoder_targets = tf.placeholder(
			shape=(None, None),
			dtype=tf.int32,
			name='decoder_targets'
		)
		self.decoder_targets_length = tf.placeholder(
			shape=(None,),
			dtype=tf.int32,
			name='decoder_targets_length',
		)

	def _init_decoder_tensors_padding_masking(self):

		with tf.name_scope('DecoderTrainFeeds'):
			sequence_size, batch_size = tf.unstack(tf.shape(self.decoder_targets))

			GO_SLICE  = tf.ones([1, batch_size], dtype=tf.int32) * data_utils.GO_ID
			PAD_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * data_utils.PAD_ID

			self.decoder_train_inputs = tf.concat([GO_SLICE, self.decoder_targets], axis=0)
			self.decoder_train_length = self.decoder_targets_length + 1

			decoder_train_targets = tf.concat([self.decoder_targets, PAD_SLICE], axis=0)
			decoder_train_targets_seq_len, _ = tf.unstack(tf.shape(decoder_train_targets))

			# create 1 hot vectors uisng lengths of the sequences in the batch
			# [batch_size, max_length+1]
			decoder_train_targets_eos_mask = tf.one_hot(self.decoder_train_length - 1,
														decoder_train_targets_seq_len,
														on_value=data_utils.EOS_ID, off_value=data_utils.EOS_ID,
														dtype=tf.int32)

			# [max_length+1, batch_size]
			decoder_train_targets_eos_mask = tf.transpose(decoder_train_targets_eos_mask, [1, 0])

			# put EOS symbol at the end of target sequence
			decoder_train_targets = tf.add(decoder_train_targets,
										   decoder_train_targets_eos_mask)

			self.decoder_train_targets = decoder_train_targets

			# mask to ignore PADs in sequences within the batch while calculating losses
			self.loss_weights = tf.transpose( tf.sign(tf.abs(self.decoder_train_targets)) )
			self.loss_weights = tf.cast(self.loss_weights, dtype=tf.float32, name="loss_weights")

	def _init_embeddings(self):

		with tf.variable_scope("embedding") as scope:

			# Uniform(-sqrt(3), sqrt(3)) has variance=1.
			sqrt3 = math.sqrt(3)
			initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

			if self.pretrain:
				print("Loading pretrained embeddings...")
				vocab, _ = data_utils.initialize_vocabulary(FLAGS.vocab_path)
				embed_matrix_np = np.random.randn(self.vocab_size, self.embedding_size).astype(np.float32)
				with open(FLAGS.embedding_file) as f:
					f.readline() # header, junk
					for line in f:
						line = line.split()
						word = line[0]
						embed = [float(i) for i in line[1:]]
						if word in vocab:
							embed_matrix_np[vocab[word]] = embed

				self.embedding_matrix = tf.Variable(embed_matrix_np)

			else:
				self.embedding_matrix = tf.get_variable(
					name="embedding_matrix",
					shape=[self.vocab_size, self.embedding_size],
					initializer=initializer,
					dtype=tf.float32)

			self.encoder_inputs_embedded = tf.nn.embedding_lookup(
					self.embedding_matrix, self.encoder_inputs)

			self.decoder_train_inputs_embedded = tf.nn.embedding_lookup(
					self.embedding_matrix, self.decoder_train_inputs)

	def _init_simple_encoder(self):
		with tf.variable_scope("Encoder") as scope:
			(self.encoder_outputs, self.encoder_state) = (
				tf.nn.dynamic_rnn(cell=self.encoder_cell,
								  inputs=self.encoder_inputs_embedded,
								  sequence_length=self.encoder_inputs_length,
								  time_major=True,
								  dtype=tf.float32)
				)

	def _init_bidirectional_encoder(self):
		with tf.variable_scope("BidirectionalEncoder") as scope:

			#encoder_outputs are independent of num_layers
			#encoder_states depend on num_layers
			((encoder_fw_outputs,
				encoder_bw_outputs),
				(encoder_fw_state,
				encoder_bw_state)) = (
				tf.nn.bidirectional_dynamic_rnn(cell_fw=self.encoder_cell,
												cell_bw=self.encoder_cell,
												inputs=self.encoder_inputs_embedded,
												sequence_length=self.encoder_inputs_length,
												time_major=True,
												dtype=tf.float32)
				)


			#self.encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)
			self.encoder_outputs = encoder_fw_outputs

			# if isinstance(encoder_fw_state, LSTMStateTuple):
			# 	encoder_state_c = tf.concat(
			# 		(encoder_fw_state.c, encoder_bw_state.c), 1, name='bidirectional_concat_c')
			# 	encoder_state_h = tf.concat(
			# 		(encoder_fw_state.h, encoder_bw_state.h), 1, name='bidirectional_concat_h')
			# 	self.encoder_state = LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)

			# elif isinstance(encoder_fw_state, tf.Tensor):
			# 	self.encoder_state = tf.concat((encoder_fw_state, encoder_bw_state), 1, name='bidirectional_concat')

			self.encoder_state = encoder_fw_state

	def _init_decoder(self):
		with tf.variable_scope("Decoder") as scope:

			def output_fn(outputs):
				return tf.contrib.layers.linear(outputs, self.vocab_size, scope=scope)

			if not self.attention:
				decoder_fn_train = seq2seq.simple_decoder_fn_train(encoder_state=self.encoder_state)
				decoder_fn_inference = seq2seq.simple_decoder_fn_inference(
					output_fn=output_fn,
					encoder_state=self.encoder_state,
					embeddings=self.embedding_matrix,
					start_of_sequence_id=data_utils.GO_ID,
					end_of_sequence_id=data_utils.EOS_ID,
					maximum_length=FLAGS.max_inf_target_len,
					num_decoder_symbols=self.vocab_size,
				)
			else:

				# attention_states: size [batch_size, max_time, num_units]
				attention_states = tf.transpose(self.encoder_outputs, [1, 0, 2])

				#attention_states = tf.zeros([batch_size, 1, self.decoder_hidden_units])

				(attention_keys,
				attention_values,
				attention_score_fn,
				attention_construct_fn) = seq2seq.prepare_attention(
					attention_states=attention_states,
					attention_option="bahdanau",
					num_units=self.decoder_hidden_units,
				)

				decoder_fn_train = seq2seq.attention_decoder_fn_train(
					encoder_state=self.encoder_state,
					attention_keys=attention_keys,
					attention_values=attention_values,
					attention_score_fn=attention_score_fn,
					attention_construct_fn=attention_construct_fn,
					name='attention_decoder'
				)

				decoder_fn_inference = seq2seq.attention_decoder_fn_inference(
					output_fn=output_fn,
					encoder_state=self.encoder_state,
					attention_keys=attention_keys,
					attention_values=attention_values,
					attention_score_fn=attention_score_fn,
					attention_construct_fn=attention_construct_fn,
					embeddings=self.embedding_matrix,
					start_of_sequence_id=data_utils.GO_ID,
					end_of_sequence_id=data_utils.EOS_ID,
					maximum_length= FLAGS.max_inf_target_len,
					num_decoder_symbols=self.vocab_size,
				)

			(self.decoder_outputs_train,
			 self.decoder_state_train,
			 self.decoder_context_state_train) = (
				seq2seq.dynamic_rnn_decoder(
					cell=self.decoder_cell,
					decoder_fn=decoder_fn_train,
					inputs=self.decoder_train_inputs_embedded,
					sequence_length=self.decoder_train_length,
					time_major=True,
					scope=scope,
				)
			)

			self.decoder_outputs_train = tf.nn.dropout(self.decoder_outputs_train, _keep_prob)

			self.decoder_logits_train = output_fn(self.decoder_outputs_train)

			# reusing the scope of training to use the same variables for inference
			scope.reuse_variables()

			(self.decoder_logits_inference,
			self.decoder_state_inference,
			self.decoder_context_state_inference) = (
				seq2seq.dynamic_rnn_decoder(
					cell=self.decoder_cell,
					decoder_fn=decoder_fn_inference,
					time_major=True,
					scope=scope,
				)
			)

			self.decoder_prediction_inference = tf.argmax(self.decoder_logits_inference, axis=-1, name='decoder_prediction_inference')

	def _init_optimizer(self):
		logits = tf.transpose(self.decoder_logits_train, [1, 0, 2])
		targets = tf.transpose(self.decoder_train_targets, [1, 0])
		self.loss = seq2seq.sequence_loss(logits=logits,
											targets=targets,
											weights=self.loss_weights,
											softmax_loss_function=self.loss_fn)

		self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

		self.gradients = self.optimizer.compute_gradients(self.loss)

		self.capped_gradients = [( tf.clip_by_value( grad, -self.max_gradient_norm, self.max_gradient_norm ), variable ) for 
																		grad, variable in self.gradients if grad is not None]

		self.train_op = self.optimizer.apply_gradients(self.capped_gradients)

	def make_train_inputs(self, input_seq, target_seq, input_seq_length, target_seq_length):
		return {
			self.encoder_inputs: input_seq,
			self.encoder_inputs_length: input_seq_length,
			self.decoder_targets: target_seq,
			self.decoder_targets_length: target_seq_length,
			_keep_prob: FLAGS.keep_prob
		}

	def make_inference_inputs(self, input_seq, input_seq_length=None):
		return {
			self.encoder_inputs: input_seq,
			self.encoder_inputs_length: input_seq_length,
			_keep_prob: 1.0
		}

	def save(self, sess, epoch):
		file_name = "dialogue_epoch_" + str(epoch) + ".ckpt"
		self.checkpoint_path = os.path.join(FLAGS.train_dir, file_name)
		self.saver.save(sess, self.checkpoint_path)

def make_seq2seq_model(**kwargs):
	args = dict(encoder_cell=LSTMCell(10),
				decoder_cell=LSTMCell(20),
				vocab_size=10,
				embedding_size=10,
				learning_rate=0.001,
				max_gradient_norm=5.0,
				attention=True,
				bidirectional=True)

	args.update(kwargs)
	model = Seq2SeqModel(**args)
	return model

def train(session, model, train_set, dev_set, batch_size=100):

	indices = np.arange(len(_buckets))

	for epoch in range(FLAGS.num_epochs):

		start = time.time()

		print("Epoch {0} started".format(epoch))
		sys.stdout.flush()

		np.random.shuffle(indices)

		for bucket_id in indices:

			np.random.shuffle(train_set[bucket_id])
			cur_train_set = train_set[bucket_id]

			# iterating on batches
			for b in range( len(cur_train_set)//batch_size ):

				cur_batch = cur_train_set[b*batch_size : b*batch_size + batch_size]

				encoder_inputs, decoder_inputs, enc_inputs_lengths, dec_inputs_lengths = get_batch(cur_batch, batch_size)

				fd = model.make_train_inputs(encoder_inputs, decoder_inputs, enc_inputs_lengths, dec_inputs_lengths)
				
				_, l = session.run([model.train_op, model.loss], fd)
				# print(l)
				# input("Enter!")

			print("Bucket {0} finished".format(bucket_id))
			sys.stdout.flush()

		end = time.time()
		print("Epoch {0} finished".format(epoch))
		print("Training time for epoch {0} = {1} mins".format(epoch, (end-start)/60))
		sys.stdout.flush()

		print("Stats on dev set:\n")
		sys.stdout.flush()

		for bucket_id in range(len(_buckets)):
			dev_loss = []

			cur_dev_set = dev_set[bucket_id]

			for b in range( len(cur_dev_set)//batch_size ):

				cur_batch = cur_dev_set[b*batch_size : b*batch_size + batch_size]

				encoder_inputs, decoder_inputs, enc_inputs_lengths, dec_inputs_lengths = get_batch(cur_batch, batch_size)

				fd = model.make_train_inputs(encoder_inputs, decoder_inputs, enc_inputs_lengths, dec_inputs_lengths)
				l = session.run(model.loss, fd)
				dev_loss.append(l)

			print("Average loss for bucket{0} on dev_set={1}".format(bucket_id, sum(dev_loss)/len(dev_loss) ))
			sys.stdout.flush()
	

		model.save(session, epoch)

def get_batch(cur_batch, batch_size):

	# find max_length of encoder and decoder inputs in the current batch
	enc_inputs_lengths    = [len(seq[0]) for seq in cur_batch]
	dec_inputs_lengths    = [len(seq[1]) for seq in cur_batch]
	encoder_max_length    = max(enc_inputs_lengths)
	decoder_max_length    = max(dec_inputs_lengths)

	# time major arrays, initialized with PAD values
	encoder_inputs = np.ones(shape=[encoder_max_length, batch_size], dtype=np.int32) * data_utils.PAD_ID
	decoder_inputs = np.ones(shape=[decoder_max_length, batch_size], dtype=np.int32) * data_utils.PAD_ID


	# for each pair of encoder decoder inputs in current batch
	for i, enc_dec_pair in enumerate(cur_batch):

		# for each word in encoder input
		for j, word in enumerate( list( reversed(enc_dec_pair[0]) ) ):
			encoder_inputs[j, i] = word

		# for each word in decoder input
		for j, word in enumerate(enc_dec_pair[1]):
			decoder_inputs[j, i] = word

	return encoder_inputs, decoder_inputs, enc_inputs_lengths, dec_inputs_lengths

def prepare_inf_input(cur_batch, batch_size):
	enc_inputs_lengths = [len(seq) for seq in cur_batch]
	encoder_max_length = max(enc_inputs_lengths)

	# time major arrays, initialized with PAD values
	encoder_inputs = np.ones(shape=[encoder_max_length, batch_size], dtype=np.int32) * data_utils.PAD_ID

	# for each encoder input in current batch
	for i, enc_inp in enumerate(cur_batch):

		# for each word in encoder input
		for j, word in enumerate( list( reversed(enc_inp) ) ):
			encoder_inputs[j, i] = word

	return encoder_inputs, enc_inputs_lengths

def generate_test_result(session, model, dev_set, vocabulary_path):

	print("Calculating perplexities")

	try:
		os.remove("perplexities_group22.txt")

	except OSError:
		pass

	if FLAGS.output_pred == 1:
		try:
			os.remove("predictions.txt")

		except OSError:
			pass

	vocab, rev_vocab = data_utils.initialize_vocabulary(vocabulary_path)

	perps = []

	# iterating sample by sample
	for i in range(0, len(dev_set), 64):

		if i%(64*10) == 0:
			print("{0} sentences done.".format(i))
			sys.stdout.flush()

		encoder_inputs, decoder_inputs, enc_inputs_lengths, dec_inputs_lengths = get_batch(dev_set[i:i+64], batch_size=64)
		fd = model.make_train_inputs(encoder_inputs, decoder_inputs, enc_inputs_lengths, dec_inputs_lengths)

		model.soft_out = tf.nn.softmax(model.decoder_logits_inference, dim=-1, name='softmax_inference_logits')
		preds, soft_out = session.run([model.decoder_prediction_inference, model.soft_out], fd)

		#decoder_logits_inference = [time, batch_size, vocab_size]


		# calculate probability of the 2 sentences returned as prediction
		# 2 predictions for each turn
		for k in range(0, 64, 2):

			log_sentence_probs = [0.0, 0.0]

			for i,b in enumerate([k,k+1]):

				cur_sent = preds[:, b]
				num = 0
				for time in range(len(cur_sent)):

					word = cur_sent[time]
					if word != data_utils.PAD_ID:
						log_sentence_probs[i] += math.log(soft_out[time, b, word], 2)
						num += 1
					
					if word == data_utils.EOS_ID:
						break		

				log_sentence_probs[i] /= num


			perps.append( "{0:.3f} {1:.3f}\n".format( math.pow( 2, -1*log_sentence_probs[0]), math.pow( 2, -1*log_sentence_probs[1] ) ) )


		with open("perplexities_group22.txt", "a") as fi:
			fi.write( "".join(perps) )
			fi.flush()


		perp = []
			
		# with gfile.GFile("perplexities_group22.txt", mode="a") as fi:
		# 	fi.write( "{0:.3f} {1:.3f}\n".format( math.pow( 2, -1*log_sentence_probs[0]), math.pow( 2, -1*log_sentence_probs[1] ) ) )


		# if FLAGS.output_pred == 1:
		# 	for b, pred in enumerate(np.transpose(preds)): # preds = [time, batch_size]

		# 		with gfile.GFile("predictions.txt", mode="a") as fi:
		# 			fi.write(data_utils.token_ids_to_sentence(pred, rev_vocab) + "\n\n")

	# with gfile.GFile("perplexities_group22.txt", mode="a") as fi:
	# 	fi.write( "".join(perps) )



	print("perplexity calculation done!")

def read_conversation_data(data_path,vocabulary_path):
	print("In read_conversation_data")
	counter  = 0
	vocab, rev_vocab = data_utils.initialize_vocabulary(vocabulary_path)
	print("vocab_length={0}".format(len(vocab)))

	data_set = [[] for _ in _buckets]
	bucket_lengths = [0 for _ in _buckets]

	with gfile.GFile(data_path, mode="r") as fi:
		for line in fi.readlines():
			counter += 1
			if counter % 50000 == 0:
				print("reading data line %d" % counter)

			entities = line.split("\t")

			if len(entities) == 3:
				source_1 = entities[0].strip()
				target_1 = entities[1].strip()

				source_2 = entities[1].strip()
				target_2 = entities[2].strip()

				source_1_ids = [int(x) for x in data_utils.sentence_to_token_ids(source_1,vocab)]
				target_1_ids = [int(x) for x in data_utils.sentence_to_token_ids(target_1,vocab)]
				#target_1_ids.append(data_utils.EOS_ID)

				source_2_ids = [int(x) for x in data_utils.sentence_to_token_ids(source_2,vocab)]
				target_2_ids = [int(x) for x in data_utils.sentence_to_token_ids(target_2,vocab)]
				#target_2_ids.append(data_utils.EOS_ID)

				for bucket_id, (source_size, target_size) in enumerate(_buckets):
					if len(source_1_ids) < source_size and len(target_1_ids) < target_size:
						data_set[bucket_id].append([source_1_ids, target_1_ids])
						bucket_lengths[bucket_id] += 1
						break

				for bucket_id, (source_size, target_size) in enumerate(_buckets):
					if len(source_2_ids) < source_size and len(target_2_ids) < target_size:
						data_set[bucket_id].append([source_2_ids, target_2_ids])
						bucket_lengths[bucket_id] += 1
						break

	return data_set, bucket_lengths, counter

def read_dev_set(dev_set_path,vocabulary_path):
	print("In read_dev_set")
	counter  = 0
	vocab, rev_vocab = data_utils.initialize_vocabulary(vocabulary_path)
	print("vocab_length={0}".format(len(vocab)))

	data_set = []

	with gfile.GFile(dev_set_path, mode="r") as fi:
		for line in fi.readlines():
			counter += 1
			if counter % 50000 == 0:
				print("reading data line %d" % counter)

			entities = line.split("\t")

			if len(entities) == 3:
				source_1 = entities[0].strip()
				target_1 = entities[1].strip()

				source_2 = entities[1].strip()
				target_2 = entities[2].strip()

				source_1_ids = [int(x) for x in data_utils.sentence_to_token_ids(source_1,vocab)]
				target_1_ids = [int(x) for x in data_utils.sentence_to_token_ids(target_1,vocab)]

				source_2_ids = [int(x) for x in data_utils.sentence_to_token_ids(source_2,vocab)]
				target_2_ids = [int(x) for x in data_utils.sentence_to_token_ids(target_2,vocab)]

				data_set.append([source_1_ids, target_1_ids])
				data_set.append([source_2_ids, target_2_ids])

	return data_set


if __name__ == '__main__':

	data_path         = FLAGS.data_path
	#dev_data          = FLAGS.dev_data
	vocab_path        = FLAGS.vocab_path
	batch_size        = FLAGS.batch_size
	num_layers        = FLAGS.num_layers
	learning_rate     = FLAGS.learning_rate
	max_gradient_norm = FLAGS.max_gradient_norm
	embedding_size    = FLAGS.embedding_size

	print("params passed:")
	if len(sys.argv) > 0:
		print(sys.argv)
	else:
		print("None")

	# running test on dev set
	if 'test' in sys.argv:

		if len(sys.argv) > 2:
			dev_data = sys.argv[2]

		if len(sys.argv) > 3:
			FLAGS.output_pred = int(sys.argv[3])

		dev_set = read_dev_set(dev_data , vocab_path)
		tf.reset_default_graph()

		with tf.Session() as session:

			_keep_prob = tf.placeholder(
							dtype=tf.float32,
							name='keep_prob',
						)

			def encoder_single_cell():
				return DropoutWrapper( GRUCell(FLAGS.encoder_hidden_units), input_keep_prob=_keep_prob ) 

			def decoder_single_cell():
				return DropoutWrapper( GRUCell(FLAGS.encoder_hidden_units), input_keep_prob=_keep_prob ) 

			if num_layers > 1:
				encoder_cell = MultiRNNCell([encoder_single_cell() for _ in range(num_layers)])
				decoder_cell = MultiRNNCell([decoder_single_cell() for _ in range(num_layers)])
			else:
				encoder_cell = encoder_single_cell()
				decoder_cell = decoder_single_cell()


			model = make_seq2seq_model(encoder_cell=encoder_cell,
					decoder_cell=decoder_cell,
					vocab_size=FLAGS.vocab_size,
					embedding_size=embedding_size,
					learning_rate=learning_rate,
					max_gradient_norm=max_gradient_norm,
					attention=True,
					bidirectional=True)


			try:
				print("Reading model parameters from {0}".format("./train_dir/epoch_5/dialogue_epoch_5.ckpt"))
				model.saver.restore(session, "./train_dir/epoch_5/dialogue_epoch_5.ckpt")

			except:
				print("Trained model not found. Exiting!")
				sys.exit()


			generate_test_result(session, model, dev_set, vocab_path)

	# conversation with model
	elif 'conv' in sys.argv:

		vocab, rev_vocab = data_utils.initialize_vocabulary(FLAGS.vocab_path)

		tf.reset_default_graph()

		with tf.Session() as session:

			_keep_prob = tf.placeholder(
							dtype=tf.float32,
							name='keep_prob',
						)

			def encoder_single_cell():
				return DropoutWrapper( GRUCell(FLAGS.encoder_hidden_units), input_keep_prob=_keep_prob) 

			def decoder_single_cell():
				return DropoutWrapper( GRUCell(FLAGS.encoder_hidden_units), input_keep_prob=_keep_prob) 

			if num_layers > 1:
				encoder_cell = MultiRNNCell([encoder_single_cell() for _ in range(num_layers)])
				decoder_cell = MultiRNNCell([decoder_single_cell() for _ in range(num_layers)])
			else:
				encoder_cell = encoder_single_cell()
				decoder_cell = decoder_single_cell()


			model = make_seq2seq_model(encoder_cell=encoder_cell,
					decoder_cell=decoder_cell,
					vocab_size=FLAGS.vocab_size,
					embedding_size=embedding_size,
					learning_rate=learning_rate,
					max_gradient_norm=max_gradient_norm,
					attention=True,
					bidirectional=True)

			try:
				print("Reading model parameters from {0}".format("./train_dir/epoch_5/dialogue_epoch_5.ckpt"))
				model.saver.restore(session, "./train_dir/epoch_5/dialogue_epoch_5.ckpt")

			except:
				print("Trained model not found. Exiting!")
				sys.exit()



			sys.stdout.write("> ")
			sys.stdout.flush()
			sentence = sys.stdin.readline()

			while sentence:

				token_ids = data_utils.sentence_to_token_ids(sentence, vocab)
				#print(token_ids)
			
				encoder_inputs, enc_inputs_lengths = prepare_inf_input([token_ids], 1)
				# print(encoder_inputs)
				# print(enc_inputs_lengths)
				fd = model.make_inference_inputs(encoder_inputs, enc_inputs_lengths)

				inf_outs = session.run(model.decoder_prediction_inference, fd)

				reply = []
				for out in np.transpose(inf_outs):
					
					reply.append(data_utils.token_ids_to_sentence(out, rev_vocab))

				print(reply)

				sys.stdout.write("> ")
				sys.stdout.flush()
				sentence = sys.stdin.readline()


	# training the model
	else:

		print("Training started")

		data_utils.create_vocabulary(vocab_path, data_path, FLAGS.vocab_size)
		train_set, train_bucket_lengths,_ = read_conversation_data(data_path, vocab_path)
		dev_set, dev_bucket_lengths,_     = read_conversation_data(dev_data , vocab_path)

		print("train_bucket_lengths")
		print(train_bucket_lengths)

		print("dev_bucket_lengths")
		print(dev_bucket_lengths)


		tf.reset_default_graph()

		with tf.Session() as session:

			_keep_prob = tf.placeholder(
							dtype=tf.float32,
							name='keep_prob',
						)

			def encoder_single_cell():
				return DropoutWrapper( LSTMCell(FLAGS.encoder_hidden_units), input_keep_prob=_keep_prob) 

			def decoder_single_cell():
				return DropoutWrapper( LSTMCell(FLAGS.encoder_hidden_units), input_keep_prob=_keep_prob) 

			if num_layers > 1:
				encoder_cell = MultiRNNCell([encoder_single_cell() for _ in range(num_layers)])
				decoder_cell = MultiRNNCell([decoder_single_cell() for _ in range(num_layers)])
			else:
				encoder_cell = encoder_single_cell()
				decoder_cell = decoder_single_cell()


			model = make_seq2seq_model(encoder_cell=encoder_cell,
					decoder_cell=decoder_cell,
					vocab_size=FLAGS.vocab_size,
					embedding_size=embedding_size,
					learning_rate=learning_rate,
					max_gradient_norm=max_gradient_norm,
					attention=True,
					bidirectional=True,
					pretrain=True)
		
			session.run(tf.global_variables_initializer())
			train(session, model, train_set, dev_set, batch_size=batch_size)
