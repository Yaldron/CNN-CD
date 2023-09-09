import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset

import tensorflow as tf
from scipy.stats import pearsonr

def conv_layer(prev_layer, numcore, runstep, is_training):
	strides = runstep
	conv_layer = tf.layers.conv2d(
		prev_layer, numcore, kernel_size=(4, 8), strides=strides, padding='same', use_bias=True, activation='tanh')
	conv_layer = tf.layers.batch_normalization(
		conv_layer, training=is_training)
	return conv_layer

def max_pool(x, kcore, runstep):
	return tf.nn.max_pool(x, ksize=[1, kcore, kcore, 1], strides=[1, runstep, runstep, 1], padding='SAME')


def fully_connected(prev_layer, num_units, is_training):
	layer = tf.layers.dense(prev_layer, num_units,
		use_bias=True, activation=None)
	layer = tf.layers.batch_normalization(layer, training=is_training)
	layer = tf.nn.tanh(layer)

	return layer


def fully_connected2(prev_layer, num_units, is_training):
	layer = tf.layers.dense(prev_layer, num_units,use_bias=True, activation=None)
	return layer

def init_weights(shape):
	return tf.Variable(tf.random_normal(shape, stddev=1))

def init_bias(shape):
	return tf.Variable(tf.random_uniform(shape, minval=-0.01, maxval=0.01))

def get_batch(X, Y, n_examples, batch_size):
	for batch_i in range(n_examples // batch_size):
		start = batch_i * batch_size
		end = start + batch_size
		batch_xs = X[start:end]
		batch_ys = Y[start:end]
		yield batch_xs, batch_ys
##############################################################################
import math
 
def calc_corr(a, b):
    a_avg = sum(a)/len(a)
    b_avg = sum(b)/len(b)
 
    cov_ab = sum([(x - a_avg)*(y - b_avg) for x,y in zip(a, b)])
    sq = math.sqrt(sum([(x - a_avg)**2 for x in a])*sum([(x - b_avg)**2 for x in b]))
    corr_factor = cov_ab/sq
 
    return corr_factor

##############################################################################
fsst1 = "../data/Trans/anom_detrend_ersstv5_187101-197312_187101-197312_Ham-LR.nc"
fzos1 = "../data/Trans/anom_detrend_zos_OBS_SODA2.2.4_187101-197312_Ham-LR.nc"

fsst2 = "../data/Valid/anom_detrend_ersstv5_198001-201812_Ham-LR.nc"
fzos2 = "../data/Valid/anom_detrend_sshg_GODAS_198001-201812_Ham-LR.nc"

nc1 = Dataset(fsst1)
nc2 = Dataset(fzos1)
nc3 = Dataset(fsst2)
nc4 = Dataset(fzos2)

ssta1 = nc1.variables['tos']
zosa1 = nc2.variables['zos']
ssta2 = nc3.variables['tos']
zosa2 = nc4.variables['zos']


lat = nc1.variables['lat'][:]
lon = nc1.variables['lon'][:]

nlat = lat[:].shape[0]
nlon = lon[:].shape[0]


###############  arrange the X data and Y data  ###############################
monst = int(os.environ.get("NMON"))
nstep = int(os.environ.get("NSTEP"))
nstep = nstep - 1


print('nlat', nlat)
print('nlon', nlon)

id_lat = (lat >= -5) & (lat <= 5)

ntrain = 100
nvali = 36

# (150*3,lat(121),lon(240),3)
X = np.zeros((ntrain, nlat, nlon, 6), dtype=float)
Y = np.zeros((ntrain, 13), dtype=float)  # (150*3)

# (5*3,lat(121),lon(240),3)
XT = np.zeros((nvali, nlat, nlon, 6), dtype=float)
YT = np.zeros((nvali, 13), dtype=float)  # (5*3)

for i in range(3):
	X[:, :, :, i] = ssta1[12 + monst - 1 -i:12 + monst - 1 - i + ntrain * 12:12, :, :]
	X[:, :, :, i + 3] = zosa1[12 + monst - 1 -i:12 + monst - 1 - i + ntrain * 12:12, :, :]
	XT[:, :, :, i] = ssta2[12 + monst - 1 -i:12 + monst - 1 - i + nvali * 12:12, :, :]
	XT[:, :, :, i + 3] = zosa2[12 + monst - 1 -i:12 + monst - 1 - i + nvali * 12:12, :, :]
X[:, :, :, 3:] = X[:, :, :, 3:] * 10.0
XT[:, :, :, 3:] = XT[:, :, :, 3:] * 10.0

# ===================== build the structure of CNN===================================
xs = tf.placeholder(tf.float32, [None, nlat, nlon, 6])  # 28x28
ys = tf.placeholder(tf.float32, [None, 13])
# keep prob is the parameter for drop out

num_convf = 36
num_hiddf = 50
xdim2 = int(nlon/4)
ydim2 = int(nlat/4)


w = init_weights([8, 4, 6, num_convf])
b = init_bias([num_convf])
w2 = init_weights([4, 2, num_convf, num_convf])
b2 = init_bias([num_convf])
w3 = init_weights([4, 2, num_convf, num_convf])
b3 = init_bias([num_convf])
w4 = init_weights([num_convf * xdim2 * ydim2, num_hiddf])
b4 = init_bias([num_hiddf])
w_o = init_weights([num_hiddf, 13])
b_o = init_bias([13])


l1a = tf.tanh(tf.nn.conv2d(xs, w, strides=[1, 1, 1, 1], padding='SAME') + b)
l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

l2a = tf.tanh(tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME') + b2)
l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

l3a = tf.tanh(tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding='SAME') + b3)
l3 = tf.reshape(l3a, [-1, w4.get_shape().as_list()[0]])


l4 = tf.tanh(tf.matmul(l3, w4) + b4)

prediction = tf.matmul(l4, w_o) + b_o

#--------------------------------------------------------
keep_prob = tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool)

# ==============================================================
rv_norm = tf.sqrt(tf.reduce_sum(tf.square(ys), axis=1))
pv_norm = tf.sqrt(tf.reduce_sum(tf.square(prediction), axis=1))

rv_pv   = tf.reduce_sum(tf.multiply(ys, prediction), axis=1)
cosin   = 1.0 - rv_pv / (rv_norm * pv_norm)

loss = tf.reduce_mean(cosin)


with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

saver = tf.train.Saver()   # set the container for net
##############################################################
train_obj = 'CMIP6_ens43_Cdist_OBS_trans'

for k in range(nstep, nstep + 1):
	print('the object mon is ', monst + k + 1)
	for ir in range(13):
		id_lon = (lon >= 130+ir*10) & (lon <= 150+ir*10)
		Y[:,ir] = np.mean(np.mean(ssta1[12 + monst + k:12 + monst + k +ntrain * 12:12, id_lat, id_lon], axis=1), axis=1).reshape([ntrain])
		YT[:,ir] = np.mean(np.mean(ssta2[12 + monst + k:12 + monst + k + nvali * 12:12, id_lat, id_lon], axis=1), axis=1).reshape([nvali])
	sess = tf.Session()

	dirpath = "./"+str(nstep + 1)+"p_model/"

	mo_path = "CMIP6_ens43_Cdist_" + \
		str(monst).zfill(2) + "s_" + str(nstep + 1).zfill(2) + "p_200t"
	save_path = saver.restore(sess, dirpath + mo_path + "/save_net.ckpt")

	# record cost
	cost = []
	record_step = 5

	for i in range(21):
		for batch_xs, batch_ys in get_batch(X, Y, Y.shape[0], batch_size=20):
			sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.9, is_training: True})
		if i % record_step == 0:
			print('this is the ', i, 'th train')
			sloss, yp = sess.run([loss, prediction], feed_dict={xs: XT, ys: YT, keep_prob: 1.0, is_training: False})
			print('Cdist: ', sloss)
			flog = open("./log_" + train_obj +str(k + 1).zfill(2) + ".log", 'a+')
			print('Cdist: ', sloss, file=flog)
			flog.close()

	dirpath = './' + train_obj + '_' + \
		str(monst).zfill(2) + 's_' + \
		str(k + 1).zfill(2) + 'p_' + str(i).zfill(3) + 't'

	if (not os.path.isdir(dirpath)):
		os.makedirs(dirpath)
	save_path = saver.save(sess, dirpath + "/save_net.ckpt")
	print("Save to path: ", save_path)

	np.savetxt('ss_obs_'+str(monst).zfill(2) + 's_' + str(k + 1).zfill(2) + 'p_' + str(i).zfill(3) + 't.txt',YT[:],fmt="%0.5f")
	np.savetxt('ss_pre_'+str(monst).zfill(2) + 's_' + str(k + 1).zfill(2) + 'p_' + str(i).zfill(3) + 't.txt',yp[:],fmt="%0.5f")
	sess.close()
