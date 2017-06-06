# Working with TF commit 24466c2e6d32621cd85f0a78d47df6eed2c5c5a6

import math
import sys

import numpy as np
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.layers import safe_embedding_lookup_sparse as embedding_lookup_unique
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell
from tensorflow.python.platform import gfile

import helpers

import data_utils


tf.app.flags.DEFINE_float("learning_rate"             , 0.5   , "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99  , "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm"         , 5.0   , "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size"              , 4    , "Batch size to use during training.")
tf.app.flags.DEFINE_integer("encoder_hidden_units"    , 32  , "Size of each model layer.")# number of dimensions in embedding space also same
tf.app.flags.DEFINE_integer("num_layers"              , 3     , "Number of layers in the model.")
tf.app.flags.DEFINE_integer("vocab_size"              , 5000  , "English vocabulary size.")
tf.app.flags.DEFINE_integer("num_epochs"              , 20    , "Number of epochs to run")
tf.app.flags.DEFINE_integer("steps_per_checkpoint"    , 200   , "How many training steps to do per checkpoint.")

tf.app.flags.DEFINE_string("data_path" , "data/Training_Shuffled_Dataset.txt", "Data directory")#done
tf.app.flags.DEFINE_string("vocab_path", "data/Vocab_file.txt", "Data directory")#done
tf.app.flags.DEFINE_string("dev_data"  , "data/Validation_Shuffled_Dataset.txt", "Data directory")#done
tf.app.flags.DEFINE_string("train_dir" , "train_dir/", "Training directory.")#done

FLAGS = tf.app.flags.FLAGS

_buckets = [(5, 10), (10, 15), (20, 25), (40, 50), (100, 100)]

class Seq2SeqModel():
	"""Seq2Seq model usign blocks from new `tf.contrib.seq2seq`.
	Requires TF 1.0.0-alpha"""

	PAD = 0
	EOS = 1
	GO  = 2
	UNK = 3

	def __init__(self, encoder_cell, decoder_cell, vocab_size, embedding_size,
				 bidirectional=True,
				 attention=False,
				 debug=False):
		self.debug = debug
		self.bidirectional = bidirectional
		self.attention = attention

		self.vocab_size = vocab_size
		self.embedding_size = embedding_size

		self.encoder_cell = encoder_cell
		self.decoder_cell = decoder_cell

		self._make_graph()

	@property
	def decoder_hidden_units(self):
		# @TODO: is this correct for LSTMStateTuple?
		return self.decoder_cell.output_size

	def _make_graph(self):
		if self.debug:
			self._init_debug_inputs()
		else:
			self._init_placeholders()

		self._init_decoder_tensors_padding_masking()
		self._init_embeddings()

		if self.bidirectional:
			self._init_bidirectional_encoder()
		else:
			self._init_simple_encoder()

		self._init_decoder()

		self._init_optimizer()

	def _init_debug_inputs(self):
		""" Everything is time-major """
		x = [[5, 6, 7],
			 [7, 6, 0],
			 [0, 7, 0]]
		xl = [2, 3, 1]
		self.encoder_inputs = tf.constant(x, dtype=tf.int32, name='encoder_inputs')
		self.encoder_inputs_length = tf.constant(xl, dtype=tf.int32, name='encoder_inputs_length')

		self.decoder_targets = tf.constant(x, dtype=tf.int32, name='decoder_targets')
		self.decoder_targets_length = tf.constant(xl, dtype=tf.int32, name='decoder_targets_length')

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
		"""
		During training, `decoder_targets`
		and decoder logits. This means that their shapes should be compatible.

		Here we do a bit of plumbing to set this up.
		"""
		with tf.name_scope('DecoderTrainFeeds'):
			sequence_size, batch_size = tf.unstack(tf.shape(self.decoder_targets))

			GO_SLICE  = tf.ones([1, batch_size], dtype=tf.int32) * self.GO
			EOS_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.EOS
			PAD_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.PAD

			self.decoder_train_inputs = tf.concat([GO_SLICE, self.decoder_targets], axis=0)
			self.decoder_train_length = self.decoder_targets_length + 1

			decoder_train_targets = tf.concat([self.decoder_targets, PAD_SLICE], axis=0)
			decoder_train_targets_seq_len, _ = tf.unstack(tf.shape(decoder_train_targets))

			# create 1 hot vectors uisng lengths of the sequences in the batch
			# [batch_size, max_length+1]
			decoder_train_targets_eos_mask = tf.one_hot(self.decoder_train_length - 1,
														decoder_train_targets_seq_len,
														on_value=self.EOS, off_value=self.PAD,
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

			# self.loss_weights = tf.ones([
			# 	batch_size,
			# 	tf.reduce_max(self.decoder_train_length)
			# ], dtype=tf.float32, name="loss_weights")

	def _init_embeddings(self):
		with tf.variable_scope("embedding") as scope:

			# Uniform(-sqrt(3), sqrt(3)) has variance=1.
			sqrt3 = math.sqrt(3)
			initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

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

			self.encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

			if isinstance(encoder_fw_state, LSTMStateTuple):

				encoder_state_c = tf.concat(
					(encoder_fw_state.c, encoder_bw_state.c), 1, name='bidirectional_concat_c')
				encoder_state_h = tf.concat(
					(encoder_fw_state.h, encoder_bw_state.h), 1, name='bidirectional_concat_h')
				self.encoder_state = LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)

			elif isinstance(encoder_fw_state, tf.Tensor):
				self.encoder_state = tf.concat((encoder_fw_state, encoder_bw_state), 1, name='bidirectional_concat')

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
					start_of_sequence_id=self.GO,
					end_of_sequence_id=self.EOS,
					maximum_length=tf.reduce_max(self.decoder_train_length),
					num_decoder_symbols=self.vocab_size,
				)
			else:

				# attention_states: size [batch_size, max_time, num_units]
				attention_states = tf.transpose(self.encoder_outputs, [1, 0, 2])

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
					start_of_sequence_id=self.GO,
					end_of_sequence_id=self.EOS,
					maximum_length=tf.reduce_max(self.decoder_train_length),
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

			self.decoder_logits_train = output_fn(self.decoder_outputs_train)
			self.decoder_prediction_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_prediction_train')

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
		self.loss = seq2seq.sequence_loss(logits=logits, targets=targets,
										  weights=self.loss_weights)
		self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

	# def make_train_inputs(self, input_seq, target_seq):
	#     inputs_, inputs_length_ = helpers.batch(input_seq)
	#     targets_, targets_length_ = helpers.batch(target_seq)
	#     return {
	#         self.encoder_inputs: inputs_,
	#         self.encoder_inputs_length: inputs_length_,
	#         self.decoder_targets: targets_,
	#         self.decoder_targets_length: targets_length_,
	#     }

	def make_train_inputs(self, input_seq, target_seq, input_seq_length, target_seq_length):
		return {
			self.encoder_inputs: input_seq,
			self.encoder_inputs_length: input_seq_length,
			self.decoder_targets: target_seq,
			self.decoder_targets_length: target_seq_length,
		}

	# def make_inference_inputs(self, input_seq):
	#     inputs_, inputs_length_ = helpers.batch(input_seq)
	#     return {
	#         self.encoder_inputs: inputs_,
	#         self.encoder_inputs_length: inputs_length_,
	#     }

	def make_inference_inputs(self, input_seq, input_seq_length):
		return {
			self.encoder_inputs: inputs_,
			self.encoder_inputs_length: inputs_length_,
		}

def make_seq2seq_model(**kwargs):
	args = dict(encoder_cell=LSTMCell(10),
				decoder_cell=LSTMCell(20),
				vocab_size=10,
				embedding_size=10,
				attention=True,
				bidirectional=True,
				debug=False)
	args.update(kwargs)
	return Seq2SeqModel(**args)

def train_on_copy_task(session, model, length_from=3, length_to=8, vocab_lower=2, vocab_upper=10, batch_size=100, max_batches=5000, batches_in_epoch=5, verbose=True):

	batches = helpers.random_sequences(length_from=length_from, length_to=length_to,
									   vocab_lower=vocab_lower, vocab_upper=vocab_upper,
									   batch_size=batch_size)
	loss_track = []
	try:
		for batch in range(max_batches+1):
			batch_data = next(batches)
			fd = model.make_train_inputs(batch_data, batch_data)
			_, l = session.run([model.train_op, model.loss], fd)
			loss_track.append(l)

			if verbose:
				if batch == 0 or batch % batches_in_epoch == 0:
					print('batch {}'.format(batch))
					print('  minibatch loss: {}'.format(session.run(model.loss, fd)))
					for i, (e_in, dt_pred) in enumerate(zip(
							fd[model.encoder_inputs].T,
							session.run(model.decoder_prediction_train, fd).T
						)):
						print('  sample {}:'.format(i + 1))
						print('    enc input           > {}'.format(e_in))
						print('    dec train predicted > {}'.format(dt_pred))
						if i >= 2:
							break
					print()
	except KeyboardInterrupt:
		print('training interrupted')

	return loss_track

def train(session, model, train_set, dev_set, batch_size=100):

	loss_track = []
	try:
		indices = np.arange(len(_buckets))

		for epoch in range(FLAGS.num_epochs):

			print("Epoch {0}".format(epoch))

			np.random.shuffle(indices)
			print("indices:")
			print(indices)

			for bucket_id in indices:
				print("bucket_id")
				print(bucket_id)
				np.random.shuffle(train_set[bucket_id])
				cur_train_set = train_set[bucket_id]

				# iterating on batches
				for b in range( len(cur_train_set)//batch_size ):

					cur_batch = cur_train_set[b*batch_size : b*batch_size + batch_size]

					encoder_inputs, decoder_inputs, enc_inputs_lengths, dec_inputs_lengths = get_batch(cur_batch, batch_size)
					print("get_batch done")

					fd = model.make_train_inputs(encoder_inputs, decoder_inputs, enc_inputs_lengths, dec_inputs_lengths)
					print("make_train_inputs done")
					
					_, l = session.run([model.train_op, model.loss], fd)
					print("loss={0}".format(l))
					input("Enter!")
					loss_track.append(l)

			
			print("Stats on dev set:\n")
			dev_loss = []
			for bucket_id in indices:
				
				cur_dev_set = dev_set[bucket_id]

				for b in range( len(cur_dev_set)//batch_size ):

					cur_batch = cur_dev_set[b*batch_size : b*batch_size + batch_size]

					encoder_inputs, decoder_inputs, enc_inputs_lengths, dec_inputs_lengths = get_batch(cur_batch, batch_size)

					fd = model.make_train_inputs(encoder_inputs, decoder_inputs, enc_inputs_lengths, dec_inputs_lengths)
					l = session.run([model.loss], fd)
					dev_loss.append(l)

			print("Average loss on dev_set={0}", sum(dev_loss)/len(dev_loss))

	except KeyboardInterrupt:
		print('training interrupted')

	return loss_track

def get_batch(cur_batch, batch_size):

	# find max_length of encoder and decoder inputs in the current batch
	enc_inputs_lengths    = [len(seq[0]) for seq in cur_batch]
	dec_inputs_lengths    = [len(seq[1]) for seq in cur_batch]
	encoder_max_length    = max(enc_inputs_lengths)
	decoder_max_length    = max(dec_inputs_lengths)

	print("encoder_max_length")
	print(encoder_max_length)

	print("decoder_max_length")
	print(decoder_max_length)

	#input("Enter!")

	# time major arrays, initialized with PAD values
	encoder_inputs = np.ones(shape=[encoder_max_length, batch_size], dtype=np.int32) * data_utils.PAD_ID
	decoder_inputs = np.ones(shape=[decoder_max_length, batch_size], dtype=np.int32) * data_utils.PAD_ID

	for i, seq in enumerate(cur_batch):
		for j, element in enumerate(seq):
			encoder_inputs[j, i] = element[0]
			decoder_inputs[j, i] = element[1]

	print("enc_inputs_lengths")
	print(enc_inputs_lengths)

	print("dec_inputs_lengths")
	print(dec_inputs_lengths)

	return encoder_inputs, decoder_inputs, enc_inputs_lengths, dec_inputs_lengths

def read_conversation_data(data_path,vocabulary_path):
	print("In read_conversation_data")
	counter  = 0
	vocab, _ = data_utils.initialize_vocabulary(vocabulary_path)
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


if __name__ == '__main__':

	tracks = {}
	print("Training")
	data_path  = FLAGS.data_path
	dev_data   = FLAGS.dev_data
	vocab_path = FLAGS.vocab_path
	batch_size = FLAGS.batch_size

	data_utils.create_vocabulary(vocab_path, data_path, FLAGS.vocab_size)
	train_set, train_bucket_lengths,_ = read_conversation_data(data_path, vocab_path)
	dev_set, dev_bucket_lengths,_     = read_conversation_data(dev_data , vocab_path)

	print("train_bucket_lengths")
	print(train_bucket_lengths)

	print("dev_bucket_lengths")
	print(dev_bucket_lengths)


	tf.reset_default_graph()

	with tf.Session() as session:
		model = make_seq2seq_model(encoder_cell=LSTMCell(FLAGS.encoder_hidden_units),
				decoder_cell=LSTMCell(2 * FLAGS.encoder_hidden_units),
				vocab_size=FLAGS.vocab_size,
				embedding_size=FLAGS.encoder_hidden_units,
				attention=True,
				bidirectional=True,
				debug=False)

		session.run(tf.global_variables_initializer())
		#loss_track_attention = train_on_copy_task(session, model)
		loss_track_attention = train(session, model, train_set, dev_set, batch_size=batch_size)












	# import sys

	# if 'fw-debug' in sys.argv:
	#     tf.reset_default_graph()
	#     with tf.Session() as session:
	#         model = make_seq2seq_model(debug=True)
	#         session.run(tf.global_variables_initializer())
	#         session.run(model.decoder_prediction_train)
	#         session.run(model.decoder_prediction_train)

	# elif 'fw-inf' in sys.argv:
	#     tf.reset_default_graph()
	#     with tf.Session() as session:
	#         model = make_seq2seq_model()
	#         session.run(tf.global_variables_initializer())
	#         fd = model.make_inference_inputs([[5, 4, 6, 7], [6, 6]])
	#         inf_out = session.run(model.decoder_prediction_inference, fd)
	#         print(inf_out)

	# elif 'train' in sys.argv:
	#     tracks = {}

	#     print("Training")
		# data_path  = FLAGS.data_path
		# dev_data   = FLAGS.dev_data
		# vocab_path = FLAGS.vocab_path

		# create_vocabulary(vocab_path, data_path, FLAGS.en_vocab_size)

		# train_set = read_conversation_data(data_path, vocab_path, FLAGS.max_train_data_size)
	# 	dev_set   = read_conversation_data(dev_data , vocab_path, FLAGS.max_train_data_size)

	#     tf.reset_default_graph()

	#     with tf.Session() as session:
	#         model = make_seq2seq_model(attention=True)
	#         session.run(tf.global_variables_initializer())
	#         loss_track_attention = train_on_copy_task(session, model)

	#     tf.reset_default_graph()

	#     with tf.Session() as session:
	#         model = make_seq2seq_model(attention=False)
	#         session.run(tf.global_variables_initializer())
	#         loss_track_no_attention = train_on_copy_task(session, model)

	#     import matplotlib.pyplot as plt
	#     plt.plot(loss_track)
	#     print('loss {:.4f} after {} examples (batch_size={})'.format(loss_track[-1], len(loss_track)*batch_size, batch_size))

	# else:
	#     tf.reset_default_graph()
	#     session = tf.InteractiveSession()
	#     model = make_seq2seq_model(debug=False)
	#     session.run(tf.global_variables_initializer())

	#     fd = model.make_inference_inputs([[5, 4, 6, 7], [6, 6]])

	#     inf_out = session.run(model.decoder_prediction_inference, fd)