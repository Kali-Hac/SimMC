import numpy as np
import tensorflow as tf
import os, sys
from utils import process_L3 as process
from utils.faiss_rerank import compute_jaccard_distance
from tensorflow.python.layers.core import Dense
from sklearn.preprocessing import label_binarize
from sklearn.cluster import DBSCAN
import torch
import collections
from sklearn.metrics import average_precision_score
from sklearn import metrics as mr
from sklearn.metrics.cluster import adjusted_mutual_info_score as AMI_score
import gc

dataset = ''
probe = ''
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

nb_nodes = 20
ft_size = 3         # originial node feature dimension (D)
time_step = 6       # sequence length (f)

# training params
batch_size = 256
nb_epochs = 100000
patience = 250     # patience for early stopping
# only raw skeletons
# hid_units = [3]  # numbers of hidden units per each attention head in each layer
# Ms = [8, 1]  # additional entry for the output layer
k1, k2 = 20, 6  # parameters to compute feature distance matrix

tf.app.flags.DEFINE_string('H', '256', "") # embedding size for each skeleton
tf.app.flags.DEFINE_string('D', '1', "") # number of hidden layers in MLP encoder
tf.app.flags.DEFINE_string('mask_x', '2', "mask some frames") # number of random masks in MPC
tf.app.flags.DEFINE_string('mask_lambda', 'best', "") # mask_lambda, mask_lambda * MIC_loss + (1 - mask_lambda) * MPC_loss
tf.app.flags.DEFINE_string('save_flag', '0', "") # save model metrics (top-1, top-5. top-10, mAP, MPC loss, MIC loss, MI, AMI, mACT, mRCL)
tf.app.flags.DEFINE_string('save_model', '1', "") # save best model
tf.app.flags.DEFINE_string('batch_size', '256', "")
tf.app.flags.DEFINE_string('mask_x_2', 'same', "") # number of random masks for second subsequence in MIC
tf.app.flags.DEFINE_string('model_size', '0', "") # output model size and computational complexity

tf.app.flags.DEFINE_string('dataset', 'KS20', "Dataset: IAS, KS20, BIWI, CASIA-B or KGBD")
tf.app.flags.DEFINE_string('length', '6', "4, 6, 8 or 10") # sequence length (f)
tf.app.flags.DEFINE_string('t', '0.07', "temperature for contrastive learning") # temperature for MPC
tf.app.flags.DEFINE_string('lr', '0.00035', "learning rate")
tf.app.flags.DEFINE_string('eps', '0.6', "distance parameter in DBSCAN")
tf.app.flags.DEFINE_string('min_samples', '2', "minimum sample number in DBSCAN")
tf.app.flags.DEFINE_string('probe', 'probe', "for testing probe")
tf.app.flags.DEFINE_string('gpu', '0', "GPU number")
tf.app.flags.DEFINE_string('probe_type', '', "probe.gallery") # probe and gallery setting for CASIA-B
tf.app.flags.DEFINE_string('patience', '100', "epochs for early stopping")
tf.app.flags.DEFINE_string('mode', 'Train', "Training (Train) or Evaluation (Eval)")
FLAGS = tf.app.flags.FLAGS


# check parameters
if FLAGS.dataset not in ['IAS', 'KGBD', 'KS20', 'BIWI', 'CASIA_B']:
	raise Exception('Dataset must be IAS, KGBD, KS20, BIWI or CASIA B.')
if FLAGS.dataset == 'CASIA_B':
	if FLAGS.mode == 'Eval':
		FLAGS.length = '40'
	if FLAGS.length not in ['40', '50', '60']:
		raise Exception('Length number must be 40, 50 or 60')
else:
	if FLAGS.length not in ['4', '6', '8', '10']:
		raise Exception('Length number must be 4, 6, 8 or 10')
if FLAGS.probe not in ['probe', 'Walking', 'Still', 'A', 'B']:
	raise Exception('Dataset probe must be "A" (for IAS-A), "B" (for IAS-B), "probe" (for KS20, KGBD).')
if FLAGS.mask_lambda != 'best' and (float(FLAGS.mask_lambda) < 0 or float(FLAGS.mask_lambda) > 1):
	raise Exception('Mask_lambda must be not less than 0 or not larger than 1.')
if FLAGS.mode not in ['Train', 'Eval']:
	raise Exception('Mode must be Train or Eval.')

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
dataset = FLAGS.dataset

# optimal paramters
if dataset == 'KGBD':
	FLAGS.lr = '0.00035'
	FLAGS.min_samples = '4'
	FLAGS.t = '0.06'
elif dataset == 'CASIA_B':
	FLAGS.lr = '0.00035'
	FLAGS.min_samples = '2'
	FLAGS.eps = '0.75'
	FLAGS.t = '0.075'
else:
	FLAGS.lr = '0.00035'
if dataset == 'KS20' or dataset == 'IAS':
	FLAGS.t = '0.08'
	FLAGS.eps = '0.8'
elif dataset == 'BIWI':
	FLAGS.t = '0.07'
	if FLAGS.probe == 'Walking':
		FLAGS.eps = '0.8'

eps = float(FLAGS.eps)
min_samples = int(FLAGS.min_samples)

time_step = int(FLAGS.length)
probe = FLAGS.probe
patience = int(FLAGS.patience)
batch_size = int(FLAGS.batch_size)

# not used
global_att = False
nhood = 1
residual = False
nonlinearity = tf.nn.elu

pre_dir = 'ReID_Models/'
# Customize the [directory] to save models with different hyper-parameters
change = ''
# change = '_SimMC' + '_H_' + FLAGS.H + '_mask_x_' + FLAGS.mask_x + '_mask_lambda_' + FLAGS.mask_lambda
# [directory] = [pre_dir] + [dataset] + '/' + [probe] + [change] + '/' + 'best.ckpt'
# e.g., ReID_Models/BIWI/Walking_SimMC_H_256_mask_x_2_mask_lambda_0.25/

if FLAGS.probe_type != '':
	change += '_CME'

try:
	os.mkdir(pre_dir)
except:
	pass

if dataset == 'KS20':
	nb_nodes = 25

if dataset == 'CASIA_B':
	nb_nodes = 14

if FLAGS.mask_lambda == 'best':
	if FLAGS.dataset == 'KS20' or FLAGS.dataset == 'KGBD':
		FLAGS.mask_lambda = '0.5'
	elif FLAGS.dataset == 'IAS' and FLAGS.probe == 'A':
		FLAGS.mask_lambda = '0.75'
	elif FLAGS.dataset == 'IAS' and FLAGS.probe == 'B':
		FLAGS.mask_lambda = '0.5'
	elif FLAGS.dataset == 'BIWI' or FLAGS.dataset == 'CASIA_B':
		FLAGS.mask_lambda = '0.25'

if FLAGS.dataset == 'CASIA_B':
	FLAGS.length = '40'
	FLAGS.mask_x = '10'


print('----- Model hyperparams -----')
print('seqence_length: ' + str(time_step))
print('mask_x: ' + FLAGS.mask_x)
print('mask_lambda: ' + FLAGS.mask_lambda)
print('H: ' + FLAGS.H)
print('batch_size: ' + str(batch_size))
print('lr: ' + str(FLAGS.lr))
print('temperature: ' + FLAGS.t)
print('eps: ' + FLAGS.eps)
print('min_samples: ' + FLAGS.min_samples)
print('patience: ' + FLAGS.patience)
print('Mode: ' + FLAGS.mode)

if FLAGS.mode == 'Train':
	print('----- Dataset Information  -----')
	print('Dataset: ' + dataset)
	if dataset == 'CASIA_B':
		print('Probe.Gallery: ', FLAGS.probe_type.split('.')[0], FLAGS.probe_type.split('.')[1])
	else:
		print('Probe: ' + FLAGS.probe)

"""
 Codes from our project of SPC-MGR
 Obtain training and testing data in joint-level, part-level, body-level, and hyper-body-level.
 Generate corresponding adjacent matrix and bias.
 We only use original skeleton data (joint-level)
"""
if FLAGS.probe_type == '':
	X_train_J, _, _, _, _, y_train, X_test_J, _, _, _, _, y_test, \
	adj_J, biases_J, _, _, _, _, _, _, _, _, nb_classes = \
		process.gen_train_data(dataset=dataset, split=probe, time_step=time_step,
		                       nb_nodes=nb_nodes, nhood=nhood, global_att=global_att, batch_size=batch_size, )
	del _
	gc.collect()

else:
	from utils import process_cme_L3 as process
	X_train_J, _, _, _, _, y_train, X_test_J, _, _, _, _, y_test, \
	adj_J, biases_J, _, _, _, _, _, _, _, _, nb_classes = \
		process.gen_train_data(dataset=dataset, split=probe, time_step=time_step,
		                       nb_nodes=nb_nodes, nhood=nhood, global_att=global_att, batch_size=batch_size, PG_type=FLAGS.probe_type.split('.')[0])
	print('## [Probe].[Gallery]', FLAGS.probe_type)
	del _
	gc.collect()

all_ftr_size = int(FLAGS.H)
loaded_graph = tf.Graph()
joint_num = X_train_J.shape[2]

cluster_epochs = 15000
display = 20

if FLAGS.mode == 'Train':
	loaded_graph = tf.Graph()
	with loaded_graph.as_default():
		with tf.name_scope('Input'):
			J_in = tf.placeholder(dtype=tf.float32, shape=(batch_size * time_step, joint_num, ft_size))
			pseudo_lab_1 = tf.placeholder(dtype=tf.int32, shape=(batch_size,))
			seq_cluster_ftr_1 = tf.placeholder(dtype=tf.float32, shape=(None, all_ftr_size))
			pseudo_lab_2 = tf.placeholder(dtype=tf.int32, shape=(batch_size,))
			seq_cluster_ftr_2 = tf.placeholder(dtype=tf.float32, shape=(None, all_ftr_size))
			seq_mask = tf.placeholder(dtype=tf.float32, shape=(2, time_step))

		with tf.name_scope("SimMC"), tf.variable_scope("SimMC", reuse=tf.AUTO_REUSE):
			inputs = tf.reshape(J_in, [time_step * batch_size, -1])
			outputs = inputs
			# skip = [-1]
			for i in range(int(FLAGS.D)):
				outputs = tf.layers.dense(outputs, int(FLAGS.H), activation=tf.nn.relu)
				# if i in skip:
				# 	outputs = tf.concat([inputs, outputs], -1)
			s_rep = outputs
			s_rep = tf.layers.dense(s_rep, int(FLAGS.H), activation=None)
			# # #
			s_rep = tf.reshape(s_rep, [-1])
			optimizer = tf.train.AdamOptimizer(learning_rate=float(FLAGS.lr))
			seq_ftr = tf.reshape(s_rep, [batch_size, time_step, -1])
			C_seq = seq_ftr
			seq_ftr = tf.reduce_mean(seq_ftr, axis=1)
			seq_ftr = tf.reshape(seq_ftr, [batch_size, -1])
			mask_1 = tf.gather(seq_mask, axis=0, indices=[0])
			mask_2 = tf.gather(seq_mask, axis=0, indices=[1])
			mask_seq_1 = tf.boolean_mask(C_seq, tf.reshape(mask_1, [-1]), axis=1)
			mask_seq_2 = tf.boolean_mask(C_seq, tf.reshape(mask_2, [-1]), axis=1)
			mask_seq_1 = tf.reduce_mean(mask_seq_1, axis=1)
			mask_seq_2 = tf.reduce_mean(mask_seq_2, axis=1)

			def MPC(pseudo_lab, all_ftr, cluster_ftr):
				all_ftr = tf.nn.l2_normalize(all_ftr, axis=-1)
				cluster_ftr = tf.nn.l2_normalize(cluster_ftr, axis=-1)
				output = tf.matmul(all_ftr, tf.transpose(cluster_ftr))
				output /= float(FLAGS.t)
				loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=pseudo_lab, logits=output))
				return loss

			def empty_loss(b):
				return tf.zeros([1])

			MPC_loss_1 = tf.cond(tf.reduce_sum(pseudo_lab_1) > 0,
			                           lambda: MPC(pseudo_lab_1, mask_seq_1, seq_cluster_ftr_1),
			                           lambda: empty_loss(pseudo_lab_1))
			MPC_loss_2 = tf.cond(tf.reduce_sum(pseudo_lab_2) > 0,
			                             lambda: MPC(pseudo_lab_2, mask_seq_2, seq_cluster_ftr_2),
			                             lambda: empty_loss(pseudo_lab_2))
			MPC_loss = tf.reduce_mean(MPC_loss_1 + MPC_loss_2)

			def MIC(u, v):
				v = tf.stop_gradient(v)
				u = tf.nn.l2_normalize(u, axis=-1)
				v = tf.nn.l2_normalize(v, axis=-1)
				return -tf.reduce_mean(tf.reduce_sum(u*v, axis=-1))
			v1, v2 = mask_seq_1, mask_seq_2
			H = int(FLAGS.H)
			W_pred = tf.Variable(tf.random_normal([H, H]))
			b_pred  = tf.Variable(tf.zeros(shape=[H, ]))
			u1 = tf.matmul(v1 ,W_pred) + b_pred
			u2 = tf.matmul(v2 ,W_pred) + b_pred
			MIC_loss = MIC(u1, v2) / 2 + MIC(u2, v1) / 2

			SimMC_loss = float(FLAGS.mask_lambda) * MIC_loss + (1 - float(FLAGS.mask_lambda)) * MPC_loss
			train_op = optimizer.minimize(SimMC_loss)

		saver = tf.train.Saver()
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

		with tf.Session(config=config) as sess:
			sess.run(init_op)

			if FLAGS.model_size == '1':
				# compute model size (M) and computational complexity (GFLOPs)
				def stats_graph(graph):
					flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
					params = tf.profiler.profile(graph,
					                             options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
					print('FLOPs: {} GFLOPS;    Trainable params: {} M'.format(flops.total_float_ops / 1e9,
					                                                           params.total_parameters / 1e6))
				stats_graph(loaded_graph)
				exit()

			mask_rand_save = []
			def train_loader(X_train_J, y_train):
				global mask_rand_save
				mask_rand_save = []
				tr_step = 0
				tr_size = X_train_J.shape[0]
				train_labels_all = []
				train_features_all = []
				train_features_all_1 = []
				train_features_all_2 = []
				while tr_step * batch_size < tr_size:
					if (tr_step + 1) * batch_size > tr_size:
						break
					X_input_J = X_train_J[tr_step * batch_size:(tr_step + 1) * batch_size]
					X_input_J = X_input_J.reshape([-1, joint_num, 3])
					labels = y_train[tr_step * batch_size:(tr_step + 1) * batch_size]
					rand_choice = np.random.choice(time_step, (time_step - int(FLAGS.mask_x)),
					                               replace=False)
					mask_rand_1 = np.zeros((time_step), dtype=bool)
					mask_rand_1[rand_choice] = True
					if FLAGS.mask_x_2 == 'same':
						rand_choice = np.random.choice(time_step, (time_step - int(FLAGS.mask_x)),
						                               replace=False)
						mask_rand_2 = np.zeros((time_step), dtype=bool)
						mask_rand_2[rand_choice] = True
					else:
						rand_choice = np.random.choice(time_step, (time_step - int(FLAGS.mask_x_2)),
						                               replace=False)
						mask_rand_2 = np.zeros((time_step), dtype=bool)
						mask_rand_2[rand_choice] = True
					mask_rand = np.concatenate([mask_rand_1, mask_rand_2], axis=0)
					mask_rand = np.reshape(mask_rand, [2, time_step])
					mask_rand_save.append(mask_rand.tolist())
					[all_features, all_features_1, all_features_2] = sess.run([seq_ftr, mask_seq_1, mask_seq_2],
					                                            feed_dict={
						                                            J_in: X_input_J,
						                                            seq_mask: mask_rand,
					                                            })
					train_features_all_1.extend(all_features_1.tolist())
					train_features_all_2.extend(all_features_2.tolist())
					train_features_all.extend(all_features.tolist())
					train_labels_all.extend(labels.tolist())
					tr_step += 1

				train_features_all = np.array(train_features_all).astype(np.float32)
				train_features_all = torch.from_numpy(train_features_all)
				if FLAGS.mask_lambda != '0':
					train_features_all_1 = np.array(train_features_all_1).astype(np.float32)
					train_features_all_1 = torch.from_numpy(train_features_all_1)
					train_features_all_2 = np.array(train_features_all_2).astype(np.float32)
					train_features_all_2 = torch.from_numpy(train_features_all_2)
					return train_features_all, train_labels_all, train_features_all_1, train_features_all_2
				else:
					return train_features_all, train_labels_all

			def gal_loader(X_train_J, y_train):
				tr_step = 0
				tr_size = X_train_J.shape[0]
				gal_logits_all = []
				gal_labels_all = []
				gal_features_all = []
				embed_1_all = []
				embed_2_all = []

				while tr_step * batch_size < tr_size:
					if (tr_step + 1) * batch_size > tr_size:
						break
					X_input_J = X_train_J[tr_step * batch_size:(tr_step + 1) * batch_size]
					X_input_J = X_input_J.reshape([-1, joint_num, 3])
					labels = y_train[tr_step * batch_size:(tr_step + 1) * batch_size]
					# no mask
					rand_choice = np.array([i for i in range(time_step)])
					mask_rand_1 = np.zeros((time_step), dtype=bool)
					mask_rand_1[rand_choice] = True
					mask_rand = np.concatenate([mask_rand_1, mask_rand_1], axis=0)
					mask_rand = np.reshape(mask_rand, [2, time_step])

					[Seq_features] = sess.run([seq_ftr],
					                              feed_dict={
						                              J_in: X_input_J,
						                              seq_mask: mask_rand,
						                              })
					gal_features_all.extend(Seq_features.tolist())
					gal_labels_all.extend(labels.tolist())
					tr_step += 1

				return gal_features_all, gal_labels_all, embed_1_all, embed_2_all


			def evaluation():
				vl_step = 0
				vl_size = X_test_J.shape[0]
				pro_labels_all = []
				pro_features_all = []
				while vl_step * batch_size < vl_size:
					if (vl_step + 1) * batch_size > vl_size:
						break
					X_input_J = X_test_J[vl_step * batch_size:(vl_step + 1) * batch_size]
					X_input_J = X_input_J.reshape([-1, joint_num, 3])
					labels = y_test[vl_step * batch_size:(vl_step + 1) * batch_size]
					# no mask
					rand_choice = np.array([i for i in range(time_step)])
					mask_rand_1 = np.zeros((time_step), dtype=bool)
					mask_rand_1[rand_choice] = True
					mask_rand = np.concatenate([mask_rand_1, mask_rand_1], axis=0)
					mask_rand = np.reshape(mask_rand, [2, time_step])
					[Seq_features] = sess.run([seq_ftr],
					                              feed_dict={
						                              J_in: X_input_J,
						                              seq_mask: mask_rand,
					                              })
					pro_labels_all.extend(labels.tolist())
					pro_features_all.extend(Seq_features.tolist())
					vl_step += 1
				X = np.array(gal_features_all)
				y = np.array(gal_labels_all)
				t_X = np.array(pro_features_all)
				t_y = np.array(pro_labels_all)
				t_y = np.argmax(t_y, axis=-1)
				y = np.argmax(y, axis=-1)

				def mean_ap(distmat, query_ids=None, gallery_ids=None,
				            query_cams=None, gallery_cams=None):
					# distmat = to_numpy(distmat)
					m, n = distmat.shape
					# Fill up default values
					if query_ids is None:
						query_ids = np.arange(m)
					if gallery_ids is None:
						gallery_ids = np.arange(n)
					if query_cams is None:
						query_cams = np.zeros(m).astype(np.int32)
					if gallery_cams is None:
						gallery_cams = np.ones(n).astype(np.int32)
					# Ensure numpy array
					query_ids = np.asarray(query_ids)
					gallery_ids = np.asarray(gallery_ids)
					query_cams = np.asarray(query_cams)
					gallery_cams = np.asarray(gallery_cams)
					# Sort and find correct matches
					indices = np.argsort(distmat, axis=1)
					matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
					# Compute AP for each query
					aps = []
					if (FLAGS.probe_type == 'nm.nm' or FLAGS.probe_type == 'cl.cl' or FLAGS.probe_type == 'bg.bg'):
						for i in range(1, m):
							valid = ((gallery_ids[indices[i]] != query_ids[i]) |
							         (gallery_cams[indices[i]] != query_cams[i]))

							y_true = matches[i, valid]
							y_score = -distmat[i][indices[i]][valid]
							# discard nan
							y_score[np.isnan(y_score)] = 0
							if not np.any(y_true): continue
							aps.append(average_precision_score(y_true, y_score))
					else:
						for i in range(m):
							valid = ((gallery_ids[indices[i]] != query_ids[i]) |
							         (gallery_cams[indices[i]] != query_cams[i]))
							y_true = matches[i, valid]
							y_score = -distmat[i][indices[i]][valid]
							# discard nan
							# y_score = np.nan_to_num(y_score)
							if not np.any(y_true): continue
							aps.append(average_precision_score(y_true, y_score))
					if len(aps) == 0:
						raise RuntimeError("No valid query")
					return np.mean(aps)


				def metrics(X, y, t_X, t_y):
					# compute Euclidean distance
					if dataset != 'CASIA_B':
						a, b = torch.from_numpy(t_X), torch.from_numpy(X)
						m, n = a.size(0), b.size(0)
						a = a.view(m, -1)
						b = b.view(n, -1)
						dist_m = torch.pow(a, 2).sum(dim=1, keepdim=True).expand(m, n) + \
						         torch.pow(b, 2).sum(dim=1, keepdim=True).expand(n, m).t()
						dist_m.addmm_(1, -2, a, b.t())
						dist_m = (dist_m.clamp(min=1e-12)).sqrt()
						mAP = mean_ap(distmat=dist_m.numpy(), query_ids=t_y, gallery_ids=y)
						_, dist_sort = dist_m.sort(1)
						dist_sort = dist_sort.numpy()
					else:
						X = np.array(X)
						t_X = np.array(t_X)
						dist_m = [(np.linalg.norm(X - i, axis=1)).tolist() for i in t_X]
						dist_m = np.array(dist_m)
						mAP = mean_ap(distmat=dist_m, query_ids=t_y, gallery_ids=y)
						dist_sort = [np.argsort(np.linalg.norm(X - i, axis=1)).tolist() for i in t_X]
						dist_sort = np.array(dist_sort)

					top_1 = top_5 = top_10 = 0
					probe_num = dist_sort.shape[0]
					if (FLAGS.probe_type == 'nm.nm' or
							FLAGS.probe_type == 'cl.cl' or FLAGS.probe_type == 'bg.bg'):
						for i in range(probe_num):
							# print(dist_sort[i, :10])
							if t_y[i] in y[dist_sort[i, 1:2]]:
								top_1 += 1
							if t_y[i] in y[dist_sort[i, 1:6]]:
								top_5 += 1
							if t_y[i] in y[dist_sort[i, 1:11]]:
								top_10 += 1
					else:
						for i in range(probe_num):
							# print(dist_sort[i, :10])
							if t_y[i] in y[dist_sort[i, :1]]:
								top_1 += 1
							if t_y[i] in y[dist_sort[i, :5]]:
								top_5 += 1
							if t_y[i] in y[dist_sort[i, :10]]:
								top_10 += 1
					return mAP, top_1 / probe_num, top_5 / probe_num, top_10 / probe_num

				mAP, top_1, top_5, top_10 = metrics(X, y, t_X, t_y)
				del X, y, t_X, t_y, pro_labels_all, pro_features_all
				gc.collect()
				return mAP, top_1, top_5, top_10

			max_acc_1 = 0
			max_acc_2 = 0
			top_5_max = 0
			top_10_max = 0
			best_cluster_info_1 = [0, 0]
			best_cluster_info_2 = [0, 0]
			cur_patience = 0
			MI_1s = []
			MI_2s = []
			AMI_1s = []
			AMI_2s = []
			top_1s = []
			top_5s = []
			top_10s = []
			mAPs = []
			MIC_losses = []
			MPC_losses = []
			uni_losses = []

			uni_losses_aug_1 = []
			uni_losses_aug_2 = []
			uni_losses_clu_1 = []
			uni_losses_clu_2 = []
			ali_losses_aug = []
			ali_matrices_clu = []
			mACT = []
			mRCL = []

			if dataset == 'KGBD' or dataset == 'KS20':
				X_train_J, _, _, _, _, y_train, X_gal_J, _, _, _, _, y_gal, \
				adj_J, biases_J, _, _, _, _, _, _, _, _, nb_classes = \
					process.gen_train_data(dataset=dataset, split='gallery', time_step=time_step,
					                       nb_nodes=nb_nodes, nhood=nhood, global_att=global_att, batch_size=batch_size,
					                       )
			elif dataset == 'BIWI':
				if probe == 'Walking':
					X_train_J, _, _, _, _, y_train, X_gal_J, _, _, _, _, y_gal, \
					adj_J, biases_J, _, _, _, _, _, _, _, _, nb_classes = \
						process.gen_train_data(dataset=dataset, split='Still', time_step=time_step,
						                       nb_nodes=nb_nodes, nhood=nhood, global_att=global_att, batch_size=batch_size,
						                       )
				else:
					X_train_J, _, _, _, _, y_train, X_gal_J, _, _, _, _, y_gal, \
					adj_J, biases_J, _, _, _, _, _, _, _, _, nb_classes = \
						process.gen_train_data(dataset=dataset, split='Walking', time_step=time_step,
						                       nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
						                       batch_size=batch_size,
						                       )
			elif dataset == 'IAS':
				if probe == 'A':
					X_train_J, _, _, _, _, y_train, X_gal_J, _, _, _, _, y_gal, \
					adj_J, biases_J, _, _, _, _, _, _, _, _, nb_classes = \
						process.gen_train_data(dataset=dataset, split='B', time_step=time_step,
						                       nb_nodes=nb_nodes, nhood=nhood, global_att=global_att, batch_size=batch_size,
						                       )
				else:
					X_train_J, _, _, _, _, y_train, X_gal_J, _, _, _, _, y_gal, \
					adj_J, biases_J, _, _, _, _, _, _, _, _, nb_classes = \
						process.gen_train_data(dataset=dataset, split='A', time_step=time_step,
						                       nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
						                       batch_size=batch_size,
						                       )
			elif dataset == 'CASIA_B':
				X_train_J, _, _, _, _, y_train, X_gal_J, _, _, _, _, y_gal, \
				adj_J, biases_J, _, _, _, _, _, _, _, _, nb_classes = \
					process.gen_train_data(dataset=dataset, split=probe, time_step=time_step,
					                       nb_nodes=nb_nodes, nhood=nhood, global_att=global_att, batch_size=batch_size,
					                       PG_type=FLAGS.probe_type.split('.')[1])
			del _
			gc.collect()
			for epoch in range(cluster_epochs):
				if FLAGS.mask_lambda != '0':
					train_features_all, train_labels_all, train_features_all_1, train_features_all_2  = train_loader(X_train_J, y_train)
				else:
					train_features_all, train_labels_all = train_loader(X_train_J, y_train)
				gal_features_all, gal_labels_all, gal_embed_1_all, gal_embed_2_all = gal_loader(X_gal_J, y_gal)

				if FLAGS.save_flag == '1':
					# Compute mean intra-class tightness (mACT) and mean inter-class tightness (mRCL)
					# see "Skeleton Prototype Contrastive Learning with Multi-level Graph Relation Modeling
					# for Unsupervised Person Re-Identification" for details of above metrics
					train_features_all = train_features_all.numpy()
					labels = np.argmax(np.array(train_labels_all), axis=-1)
					label_t = set(labels.tolist())
					y = np.array(labels)
					X = np.array(train_features_all)
					sorted_indices = np.argsort(y, axis=0)
					sort_y = y[sorted_indices]
					sort_X = X[sorted_indices]
					all_class_ftrs = {}
					class_start_indices = {}
					class_end_indices = {}
					pre_label = sort_y[0]
					class_start_indices[pre_label] = 0
					for i, label in enumerate(sort_y):
						if sort_y[i] not in all_class_ftrs.keys():
							all_class_ftrs[sort_y[i]] = [sort_X[i]]
						else:
							all_class_ftrs[sort_y[i]].append(sort_X[i])
						if label != pre_label:
							class_start_indices[label] = class_end_indices[pre_label] = i
							pre_label = label
						if i == len(sort_y) - 1:
							class_end_indices[label] = i
					center_ftrs = []
					for label, class_ftrs in all_class_ftrs.items():
						class_ftrs = np.array(class_ftrs)
						center_ftr = np.mean(class_ftrs, axis=0)
						center_ftrs.append(center_ftr)
					center_ftrs = np.array(center_ftrs)

					a, b = torch.from_numpy(sort_X), torch.from_numpy(center_ftrs)

					a_norm = a / a.norm(dim=1)[:, None]
					b_norm = b / b.norm(dim=1)[:, None]
					dist_m = 1 - torch.mm(a_norm, b_norm.t())
					dist_m = dist_m.numpy()

					prototype_dis_m = np.zeros([nb_classes, nb_classes])
					for i in range(nb_classes):
						prototype_dis_m[i, :] = np.mean(dist_m[class_start_indices[i]:class_end_indices[i], :], axis=0)

					intra_class_dis = np.mean(prototype_dis_m.diagonal())
					sum_distance = np.reshape(np.sum(prototype_dis_m, axis=-1), [nb_classes, ])
					average_distance = np.sum(sum_distance) / (nb_classes * nb_classes)

					a = b = torch.from_numpy(center_ftrs)
					a_norm = a / a.norm(dim=1)[:, None]
					b_norm = b / b.norm(dim=1)[:, None]
					dist_m = 1 - torch.mm(a_norm, b_norm.t())
					dist_m = dist_m.numpy()
					inter_class_dis = np.mean(dist_m)

					mACT.append(average_distance / intra_class_dis)
					mRCL.append(inter_class_dis / average_distance)
					print ('mACT: ', average_distance / intra_class_dis, 'mRCL: ', inter_class_dis / average_distance)
					train_features_all = torch.from_numpy(train_features_all)

				mAP, top_1, top_5, top_10 = evaluation()
				cur_patience += 1
				if epoch > 0 and top_1 > max_acc_2:
					max_acc_1 = mAP
					max_acc_2 = top_1
					top_5_max = top_5
					top_10_max = top_10
					best_cluster_info_1[0] = num_cluster
					best_cluster_info_1[1] = outlier_num
					cur_patience = 0
					if FLAGS.mode == 'Train':
						if FLAGS.dataset != 'CASIA_B':
							checkpt_file = pre_dir + dataset + '/' + probe + change + '/' + 'best.ckpt'
						elif FLAGS.dataset == 'CASIA_B':
							checkpt_file = pre_dir + dataset + '/' + probe + change + '/' + FLAGS.probe_type + '_best.ckpt'
						print(checkpt_file)
						if FLAGS.save_model == '1':
							saver.save(sess, checkpt_file)
				if epoch > 0:
					print(
						'[Probe Evaluation] %s - %s | mAP: %.4f (%.4f) | Top-1: %.4f (%.4f) | Top-5: %.4f (%.4f) | Top-10: %.4f (%.4f)' % (
						FLAGS.dataset, FLAGS.probe, mAP, max_acc_1,
						top_1, max_acc_2, top_5, top_5_max, top_10, top_10_max))
				if cur_patience == patience:
					break

				def generate_cluster_features(labels, features):
					centers = collections.defaultdict(list)
					for i, label in enumerate(labels):
						if label == -1:
							continue
						centers[labels[i]].append(features[i])

					centers = [
						torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
					]
					centers = torch.stack(centers, dim=0)
					return centers

				# mask_seq_1
				rerank_dist_1 = compute_jaccard_distance(train_features_all_1, k1=k1, k2=k2)
				cluster_1 = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed', n_jobs=-1)
				pseudo_labels_1 = cluster_1.fit_predict(rerank_dist_1)

				rerank_dist_2 = compute_jaccard_distance(train_features_all_2, k1=k1, k2=k2)
				cluster_2 = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed', n_jobs=-1)
				pseudo_labels_2 = cluster_2.fit_predict(rerank_dist_2)

				cluster_features_1 = generate_cluster_features(pseudo_labels_1, train_features_all_1)
				cluster_features_1 = cluster_features_1.numpy()
				cluster_features_1 = cluster_features_1.astype(np.float64)
				cluster_features_2 = generate_cluster_features(pseudo_labels_2, train_features_all_2)
				cluster_features_2 = cluster_features_2.numpy()
				cluster_features_2 = cluster_features_2.astype(np.float64)

				# discard outliers
				X_train_J_new = X_train_J[np.where((pseudo_labels_1 != -1) & (pseudo_labels_2 != -1))]
				outlier_num = np.sum((pseudo_labels_1 == -1) | (pseudo_labels_2 == -1))
				pseudo_labels = pseudo_labels_1[np.where((pseudo_labels_1 != -1) & (pseudo_labels_2 != -1))]
				train_labels_all = np.array(train_labels_all)
				train_labels_all = train_labels_all[np.where((pseudo_labels_1 != -1) & (pseudo_labels_2 != -1))]
				pseudo_labels_2 = pseudo_labels_2[np.where((pseudo_labels_1 != -1) & (pseudo_labels_2 != -1))]
				pseudo_labels_1 = pseudo_labels

				# compute mutual information (MI) and adjusted mutual information (AMI) metrics
				y_true = np.argmax(train_labels_all, axis=-1)
				MI_1 = mr.mutual_info_score(y_true, pseudo_labels_1)
				MI_2 = mr.mutual_info_score(y_true, pseudo_labels_2)
				AMI_1 = AMI_score(y_true, pseudo_labels_1)
				AMI_2 = AMI_score(y_true, pseudo_labels_2)

				MI_1s.append(MI_1)
				MI_2s.append(MI_2)
				AMI_1s.append(AMI_1)
				AMI_2s.append(AMI_2)
				top_1s.append(top_1)
				top_5s.append(top_5)
				top_10s.append(top_10)
				mAPs.append(mAP)

				print('MI-1: %.5f | MI-2: %.5f | AMI-1: %.5f | AMI-2: %.5f |' %
							(MI_1, MI_2, AMI_1, AMI_2))

				assert len(pseudo_labels_1) == len(pseudo_labels_2)
				num_cluster = len(set(pseudo_labels_1)) - (1 if -1 in pseudo_labels_1 else 0)

				cluster_features = generate_cluster_features(pseudo_labels, train_features_all)
				cluster_features = cluster_features.numpy()
				cluster_features = cluster_features.astype(np.float64)

				tr_step = 0
				tr_size = X_train_J_new.shape[0]

				mask_rand_save = np.array(mask_rand_save)
				batch_MPC_loss = []
				batch_MIC_loss = []

				while tr_step * batch_size < tr_size:
					if (tr_step + 1) * batch_size > tr_size:
						break
					X_input_J = X_train_J_new[tr_step * batch_size:(tr_step + 1) * batch_size]
					X_input_J = X_input_J.reshape([-1, joint_num, 3])
					labels = pseudo_labels[tr_step * batch_size:(tr_step + 1) * batch_size]
					labels_1 = pseudo_labels_1[tr_step * batch_size:(tr_step + 1) * batch_size]
					labels_2 = pseudo_labels_2[tr_step * batch_size:(tr_step + 1) * batch_size]
					mask_rand = mask_rand_save[tr_step:(tr_step + 1)]
					mask_rand = np.reshape(mask_rand, [2, time_step])
					_, loss, MIC_loss_, Seq_features = sess.run(
						[train_op, MPC_loss, MIC_loss, seq_ftr],
						feed_dict={
							J_in: X_input_J,
							pseudo_lab_1: labels_1,
							pseudo_lab_2: labels_2,
							seq_cluster_ftr_1: cluster_features_1,
							seq_cluster_ftr_2: cluster_features_2,
							seq_mask: mask_rand})
					batch_MPC_loss.append(loss)
					batch_MIC_loss.append(MIC_loss_)

					if tr_step % display == 0:
						print('[%s] Batch num: %d | Cluser num: %d | Outlier: %d | MPC Loss: %.5f | MIC Loss: %.5f |' %
						      (str(epoch), tr_step, num_cluster, outlier_num, loss, MIC_loss_))
					tr_step += 1

				MPC_losses.append(np.mean(batch_MPC_loss))
				MIC_losses.append(np.mean(batch_MIC_loss))

			if FLAGS.save_flag == '1':
				try:
					os.mkdir(pre_dir + dataset +  '/' + probe + change + '/')
				except:
					pass
				np.save(pre_dir + dataset +  '/' + probe + change + '/' + 'MI_1s.npy', MI_1s)
				np.save(pre_dir + dataset +  '/' + probe + change + '/' + 'MI_2s.npy', MI_2s)
				np.save(pre_dir + dataset +  '/' + probe + change + '/' + 'AMI_1s.npy', AMI_1s)
				np.save(pre_dir + dataset +  '/' + probe + change + '/' + 'AMI_2s.npy', AMI_2s)
				np.save(pre_dir + dataset +  '/' + probe + change + '/' + 'top_1s.npy', top_1s)
				np.save(pre_dir + dataset +  '/' + probe + change + '/' + 'top_5s.npy', top_5s)
				np.save(pre_dir + dataset +  '/' + probe + change + '/' + 'top_10s.npy', top_10s)
				np.save(pre_dir + dataset +  '/' + probe + change + '/' + 'mAPs.npy', mAPs)
				np.save(pre_dir + dataset +  '/' + probe + change + '/' + 'MPC_loss.npy', MPC_losses)
				np.save(pre_dir + dataset +  '/' + probe + change + '/' + 'MIC_loss.npy', MIC_losses)
				np.save(pre_dir + dataset + '/' + probe + change + '/' + 'mACT.npy', mACT)
				np.save(pre_dir + dataset + '/' + probe + change + '/' + 'mRCL.npy', mRCL)

			sess.close()

elif FLAGS.mode == 'Eval':
	checkpt_file = pre_dir + FLAGS.dataset + '/' + FLAGS.probe + change + '/best.ckpt'

	with tf.Session(graph=loaded_graph, config=config) as sess:
		loader = tf.train.import_meta_graph(checkpt_file + '.meta')
		J_in = loaded_graph.get_tensor_by_name("Input/Placeholder:0")
		pseudo_lab_1 = loaded_graph.get_tensor_by_name("Input/Placeholder_1:0")
		seq_cluster_ftr_1 = loaded_graph.get_tensor_by_name("Input/Placeholder_2:0")
		pseudo_lab_2 = loaded_graph.get_tensor_by_name("Input/Placeholder_3:0")
		seq_cluster_ftr_2 = loaded_graph.get_tensor_by_name("Input/Placeholder_4:0")
		seq_mask = loaded_graph.get_tensor_by_name("Input/Placeholder_5:0")
		seq_ftr = loaded_graph.get_tensor_by_name("SimMC/SimMC/Mean:0")
		mask_seq_1 = loaded_graph.get_tensor_by_name("SimMC/SimMC/Mean_1:0")
		mask_seq_2 = loaded_graph.get_tensor_by_name("SimMC/SimMC/Mean_2:0")
		contrastive_loss = loaded_graph.get_tensor_by_name("SimMC/SimMC/add_4:0")
		cluster_train_op = loaded_graph.get_operation_by_name("SimMC/SimMC/Adam")
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		loader.restore(sess, checkpt_file)
		saver = tf.train.Saver()
		mask_rand_save = []
		def train_loader(X_train_J, y_train):
			global mask_rand_save
			mask_rand_save = []
			tr_step = 0
			tr_size = X_train_J.shape[0]
			train_labels_all = []
			train_features_all = []
			train_features_all_1 = []
			train_features_all_2 = []
			while tr_step * batch_size < tr_size:
				if (tr_step + 1) * batch_size > tr_size:
					break
				X_input_J = X_train_J[tr_step * batch_size:(tr_step + 1) * batch_size]
				X_input_J = X_input_J.reshape([-1, joint_num, 3])
				labels = y_train[tr_step * batch_size:(tr_step + 1) * batch_size]
				rand_choice = np.random.choice(time_step, (time_step - int(FLAGS.mask_x)),
				                               replace=False)
				mask_rand_1 = np.zeros((time_step), dtype=bool)
				mask_rand_1[rand_choice] = True
				rand_choice = np.random.choice(time_step, (time_step - int(FLAGS.mask_x)),
				                               replace=False)
				mask_rand_2 = np.zeros((time_step), dtype=bool)
				mask_rand_2[rand_choice] = True
				mask_rand = np.concatenate([mask_rand_1, mask_rand_2], axis=0)
				mask_rand = np.reshape(mask_rand, [2, time_step])
				mask_rand_save.append(mask_rand.tolist())
				[all_features, all_features_1, all_features_2] = sess.run([seq_ftr, mask_seq_1, mask_seq_2],
				                                                          feed_dict={
					                                                          J_in: X_input_J,
					                                                          seq_mask: mask_rand,
				                                                          })
				train_features_all_1.extend(all_features_1.tolist())
				train_features_all_2.extend(all_features_2.tolist())
				train_features_all.extend(all_features.tolist())
				train_labels_all.extend(labels.tolist())
				tr_step += 1

			train_features_all = np.array(train_features_all).astype(np.float32)
			train_features_all = torch.from_numpy(train_features_all)
			train_features_all_1 = np.array(train_features_all_1).astype(np.float32)
			train_features_all_1 = torch.from_numpy(train_features_all_1)
			train_features_all_2 = np.array(train_features_all_2).astype(np.float32)
			train_features_all_2 = torch.from_numpy(train_features_all_2)
			return train_features_all, train_labels_all, train_features_all_1, train_features_all_2

		def gal_loader(X_train_J, y_train):
			tr_step = 0
			tr_size = X_train_J.shape[0]
			gal_labels_all = []
			gal_features_all = []
			while tr_step * batch_size < tr_size:
				if (tr_step + 1) * batch_size > tr_size:
					break
				X_input_J = X_train_J[tr_step * batch_size:(tr_step + 1) * batch_size]
				X_input_J = X_input_J.reshape([-1, joint_num, 3])
				labels = y_train[tr_step * batch_size:(tr_step + 1) * batch_size]
				[Seq_features] = sess.run(
					[seq_ftr],
					feed_dict={
						J_in: X_input_J,
					})
				gal_features_all.extend(Seq_features.tolist())
				gal_labels_all.extend(labels.tolist())
				tr_step += 1
			return gal_features_all, gal_labels_all

		def evaluation():
			vl_step = 0
			vl_size = X_test_J.shape[0]
			pro_labels_all = []
			pro_features_all = []
			pro_features_all_1 = []
			pro_features_all_2 = []
			while vl_step * batch_size < vl_size:
				if (vl_step + 1) * batch_size > vl_size:
					break
				X_input_J = X_test_J[vl_step * batch_size:(vl_step + 1) * batch_size]
				X_input_J = X_input_J.reshape([-1, joint_num, 3])
				labels = y_test[vl_step * batch_size:(vl_step + 1) * batch_size]
				rand_choice = np.random.choice(time_step, (time_step - int(FLAGS.mask_x)),
				                               replace=False)
				mask_rand_1 = np.zeros((time_step), dtype=bool)
				mask_rand_1[rand_choice] = True
				rand_choice = np.random.choice(time_step, (time_step - int(FLAGS.mask_x)),
				                               replace=False)
				mask_rand_2 = np.zeros((time_step), dtype=bool)
				mask_rand_2[rand_choice] = True
				mask_rand = np.concatenate([mask_rand_1, mask_rand_2], axis=0)
				mask_rand = np.reshape(mask_rand, [2, time_step])
				mask_rand_save.append(mask_rand.tolist())
				[Seq_features, Seq_features_1, Seq_features_2] = sess.run([seq_ftr, mask_seq_1, mask_seq_2],
				                                                          feed_dict={
					                                                          J_in: X_input_J,
					                                                          seq_mask: mask_rand,
				                                                          })
				pro_features_all_1.extend(Seq_features_1.tolist())
				pro_features_all_2.extend(Seq_features_2.tolist())
				pro_labels_all.extend(labels.tolist())
				pro_features_all.extend(Seq_features.tolist())
				vl_step += 1
			X = np.array(gal_features_all)
			y = np.array(gal_labels_all)
			t_X = np.array(pro_features_all)
			t_X_1 = np.array(pro_features_all_1)
			t_X_2 = np.array(pro_features_all_2)
			t_y = np.array(pro_labels_all)
			t_y = np.argmax(t_y, axis=-1)
			y = np.argmax(y, axis=-1)

			def mean_ap(distmat, query_ids=None, gallery_ids=None,
			            query_cams=None, gallery_cams=None):
				# distmat = to_numpy(distmat)
				m, n = distmat.shape
				# Fill up default values
				if query_ids is None:
					query_ids = np.arange(m)
				if gallery_ids is None:
					gallery_ids = np.arange(n)
				if query_cams is None:
					query_cams = np.zeros(m).astype(np.int32)
				if gallery_cams is None:
					gallery_cams = np.ones(n).astype(np.int32)
				# Ensure numpy array
				query_ids = np.asarray(query_ids)
				gallery_ids = np.asarray(gallery_ids)
				query_cams = np.asarray(query_cams)
				gallery_cams = np.asarray(gallery_cams)
				# Sort and find correct matches
				indices = np.argsort(distmat, axis=1)
				matches = (gallery_ids[indices] == query_ids[:, np.newaxis])

				# Compute AP for each query
				aps = []
				if (FLAGS.probe_type == 'nm.nm' or FLAGS.probe_type == 'cl.cl' or FLAGS.probe_type == 'bg.bg'):
					for i in range(1, m):
						valid = ((gallery_ids[indices[i]] != query_ids[i]) |
						         (gallery_cams[indices[i]] != query_cams[i]))

						y_true = matches[i, valid]
						y_score = -distmat[i][indices[i]][valid]
						# discard nan
						y_score[np.isnan(y_score)] = 0
						if not np.any(y_true): continue
						aps.append(average_precision_score(y_true, y_score))
				else:
					for i in range(m):
						valid = ((gallery_ids[indices[i]] != query_ids[i]) |
						         (gallery_cams[indices[i]] != query_cams[i]))
						y_true = matches[i, valid]
						y_score = -distmat[i][indices[i]][valid]
						# discard nan
						# y_score = np.nan_to_num(y_score)
						if not np.any(y_true): continue
						aps.append(average_precision_score(y_true, y_score))
				if len(aps) == 0:
					raise RuntimeError("No valid query")
				return np.mean(aps)

			def metrics(X, y, t_X, t_y):
				# compute Euclidean distance
				if dataset != 'CASIA_B':
					a, b = torch.from_numpy(t_X), torch.from_numpy(X)
					m, n = a.size(0), b.size(0)
					a = a.view(m, -1)
					b = b.view(n, -1)
					# print(np.min(a.numpy()), np.max(a.numpy()), np.min(b.numpy()), np.max(b.numpy()))
					# exit()
					# print(len(a.numpy()[np.isnan(a.numpy())]), len(b.numpy()[np.isnan(b.numpy())]))
					dist_m = torch.pow(a, 2).sum(dim=1, keepdim=True).expand(m, n) + \
					         torch.pow(b, 2).sum(dim=1, keepdim=True).expand(n, m).t()
					# print(len(dist_m.numpy()[np.isnan(dist_m.numpy())]))
					dist_m.addmm_(1, -2, a, b.t())
					# print(len(dist_m.numpy()[np.isnan(dist_m.numpy())]))
					# exit()
					# print(np.min(dist_m.numpy()), np.max(dist_m.numpy()))
					dist_m = (dist_m).sqrt()
					# dist_m = (dist_m+1e-12).sqrt()
					# print(len(dist_m.numpy()[np.isnan(dist_m.numpy())]))
					# print(dist_m.numpy())
					mAP = mean_ap(distmat=dist_m.numpy(), query_ids=t_y, gallery_ids=y)
					_, dist_sort = dist_m.sort(1)
					dist_sort = dist_sort.numpy()
				else:
					X = np.array(X)
					t_X = np.array(t_X)
					# pred = [cp.argmin(cp.linalg.norm(X - i, axis=1)).tolist() for i in t_X]
					dist_m = [(np.linalg.norm(X - i, axis=1)).tolist() for i in t_X]
					dist_m = np.array(dist_m)
					mAP = mean_ap(distmat=dist_m, query_ids=t_y, gallery_ids=y)
					dist_sort = [np.argsort(np.linalg.norm(X - i, axis=1)).tolist() for i in t_X]
					dist_sort = np.array(dist_sort)

				top_1 = top_5 = top_10 = 0
				probe_num = dist_sort.shape[0]
				if (FLAGS.probe_type == 'nm.nm' or FLAGS.probe_type == 'cl.cl' or FLAGS.probe_type == 'bg.bg'):
					for i in range(probe_num):
						# print(dist_sort[i, :10])
						if t_y[i] in y[dist_sort[i, 1:2]]:
							top_1 += 1
						if t_y[i] in y[dist_sort[i, 1:6]]:
							top_5 += 1
						if t_y[i] in y[dist_sort[i, 1:11]]:
							top_10 += 1
				else:
					for i in range(probe_num):
						# print(dist_sort[i, :10])
						if t_y[i] in y[dist_sort[i, :1]]:
							top_1 += 1
						if t_y[i] in y[dist_sort[i, :5]]:
							top_5 += 1
						if t_y[i] in y[dist_sort[i, :10]]:
							top_10 += 1
				return mAP, top_1 / probe_num, top_5 / probe_num, top_10 / probe_num

			# mAP, top_1, top_5, top_10 = metrics(X, y, t_X_1, t_y)
			# print('Rand. Masked Seq 1 mAP: ', mAP, 'top-1: ', top_1, 'top-5: ', top_5, 'top-10: ', top_10)
			# mAP, top_1, top_5, top_10 = metrics(X, y, t_X_2, t_y)
			# print('Rand. Masked Seq 2 mAP: ', mAP, 'top-1: ', top_1, 'top-5: ', top_5, 'top-10: ', top_10)
			mAP, top_1, top_5, top_10 = metrics(X, y, t_X, t_y)
			del X, y, t_X, t_y, pro_labels_all, pro_features_all
			gc.collect()
			return mAP, top_1, top_5, top_10

		max_acc_1 = 0
		max_acc_2 = 0
		best_cluster_info_1 = [0, 0]
		best_cluster_info_2 = [0, 0]
		cur_patience = 0

		if dataset == 'KGBD' or dataset == 'KS20':
			X_train_J, _, _, _, _, y_train, X_gal_J, _, _, _, _, y_gal, \
			adj_J, biases_J, _, _, _, _, _, _, _, _, nb_classes = \
				process.gen_train_data(dataset=dataset, split='gallery', time_step=time_step,
				                       nb_nodes=nb_nodes, nhood=nhood, global_att=global_att, batch_size=batch_size,
				                       )
		elif dataset == 'BIWI':
			if probe == 'Walking':
				X_train_J, _, _, _, _, y_train, X_gal_J, _, _, _, _, y_gal, \
				adj_J, biases_J, _, _, _, _, _, _, _, _, nb_classes = \
					process.gen_train_data(dataset=dataset, split='Still', time_step=time_step,
					                       nb_nodes=nb_nodes, nhood=nhood, global_att=global_att, batch_size=batch_size,
					                       )
			else:
				X_train_J, _, _, _, _, y_train, X_gal_J, _, _, _, _, y_gal, \
				adj_J, biases_J, _, _, _, _, _, _, _, _, nb_classes = \
					process.gen_train_data(dataset=dataset, split='Walking', time_step=time_step,
					                       nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
					                       batch_size=batch_size,
					                       )
		elif dataset == 'IAS':
			if probe == 'A':
				X_train_J, _, _, _, _, y_train, X_gal_J, _, _, _, _, y_gal, \
				adj_J, biases_J, _, _, _, _, _, _, _, _, nb_classes = \
					process.gen_train_data(dataset=dataset, split='B', time_step=time_step,
					                       nb_nodes=nb_nodes, nhood=nhood, global_att=global_att, batch_size=batch_size,
					                       )
			else:
				X_train_J, _, _, _, _, y_train, X_gal_J, _, _, _, _, y_gal, \
				adj_J, biases_J, _, _, _, _, _, _, _, _, nb_classes = \
					process.gen_train_data(dataset=dataset, split='A', time_step=time_step,
					                       nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
					                       batch_size=batch_size,
					                       )
		elif dataset == 'CASIA_B':
			X_train_J, _, _, _, _, y_train, X_gal_J, _, _, _, _, y_gal, \
			adj_J, biases_J, _, _, _, _, _, _, _, _, nb_classes = \
				process.gen_train_data(dataset=dataset, split=probe, time_step=time_step,
				                       nb_nodes=nb_nodes, nhood=nhood, global_att=global_att, batch_size=batch_size,
				                       PG_type=FLAGS.probe_type.split('.')[1])

		mAP_max = top_1_max = top_5_max = top_10_max = 0
		gal_features_all, gal_labels_all = gal_loader(X_gal_J, y_gal)
		mAP, top_1, top_5, top_10 = evaluation()
		print(
			'[Evaluation on %s - %s] mAP: %.4f | Acc: %.4f - %.4f - %.4f |' %
			(FLAGS.dataset, FLAGS.probe, mAP, top_1, top_5, top_10,))
		sess.close()
		exit()


print('----- Model hyperparams -----')
print('seqence_length: ' + str(time_step))
print('mask_x: ' + FLAGS.mask_x)
print('mask_lambda: ' + FLAGS.mask_lambda)
print('H: ' + FLAGS.H)
print('batch_size: ' + str(batch_size))
print('lr: ' + str(FLAGS.lr))
print('temperature: ' + FLAGS.t)
print('eps: ' + FLAGS.eps)
print('min_samples: ' + FLAGS.min_samples)
print('patience: ' + FLAGS.patience)


if FLAGS.mode == 'Train':
	print('----- Dataset Information  -----')
	print('Dataset: ' + dataset)
	print('Probe: ' + FLAGS.probe)


