import os
import numpy as np
from netCDF4 import Dataset
import xarray as xr
import tensorflow as tf

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

#------------------------------------------------------------------------------------------------------
fsst2 = "/data3/home/sunming/ML/CNN/prepare_data_Ham-LR/OBS_test/anom_ersstv5_198001-201812_Ham-LR.nc"
fzos2 = "/data3/home/sunming/ML/CNN/prepare_data_Ham-LR/OBS_test/anom_sshg_GODAS_198001-201812_Ham-LR.nc"

nc3 = Dataset(fsst2)
nc4 = Dataset(fzos2)

ssta2 = nc3.variables['tos']
zosa2 = nc4.variables['zos']

lat = nc3.variables['lat'][:]
lon = nc3.variables['lon'][:]

nlat = lat[:].shape[0]
nlon = lon[:].shape[0]

#-------prepare input data-----------------------------
monst = 4 #int(os.environ.get("NMON"))
nstep = 10 #int(os.environ.get("NSTEP"))
nstep = nstep - 1

id_lat = (lat >= -5) & (lat <= 5)
nvali = 36

XT = np.zeros((nvali, nlat, nlon, 6), dtype=float)
YT = np.zeros((nvali, 13), dtype=float)  # (5*3)

for i in range(3):
	XT[:, :, :, i] = ssta2[12 + monst - 1 -i:12 + monst - 1 - i + nvali * 12:12, :, :]
	XT[:, :, :, i + 3] = zosa2[12 + monst - 1 -i:12 + monst - 1 - i + nvali * 12:12, :, :]
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


rv_norm = tf.sqrt(tf.reduce_sum(tf.square(ys), axis=1))
pv_norm = tf.sqrt(tf.reduce_sum(tf.square(prediction), axis=1))

rv_pv   = tf.reduce_sum(tf.multiply(ys, prediction), axis=1)
cosin   = 1.0 - rv_pv / (rv_norm * pv_norm)

loss = tf.reduce_mean(cosin)

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)


saver = tf.train.Saver()
#----------------------------------------------------------
print('the object mon is ', monst + nstep + 1)
sess = tf.Session()
#case_num = int(os.environ.get("CASE_NUM"))

dirpath = "./"+str(nstep + 1)+"p_model/"
mo_path = "CMIP6_ens43_Cdist_OBS_trans_" + str(monst).zfill(2) + "s_" + str(nstep + 1).zfill(2) + "p_020t"
saver.restore(sess, dirpath + mo_path + "/save_net.ckpt")

heat_mean = np.zeros((nvali,13,ydim2,xdim2))

for k in range(nvali):
#for k in [case_num]:
    conv1 = sess.run(l3a, feed_dict={xs: XT[k,:,:,:].reshape(1,nlat,nlon,6),keep_prob: 1.0, is_training: False})
    conv2 = sess.run(w4,  feed_dict={xs: XT[k,:,:,:].reshape(1,nlat,nlon,6),keep_prob: 1.0, is_training: False})
    conv3 = sess.run(w_o, feed_dict={xs: XT[k,:,:,:].reshape(1,nlat,nlon,6),keep_prob: 1.0, is_training: False})
    conv4 = sess.run(b4,  feed_dict={xs: XT[k,:,:,:].reshape(1,nlat,nlon,6),keep_prob: 1.0, is_training: False})
    conv5 = sess.run(b_o, feed_dict={xs: XT[k,:,:,:].reshape(1,nlat,nlon,6),keep_prob: 1.0, is_training: False}) 
    mul_w = np.zeros((ydim2,xdim2,num_hiddf))
    conv1 = conv1.reshape(ydim2,xdim2,num_convf)
    conv2 = conv2.reshape(ydim2,xdim2,num_convf,num_hiddf)
    conv3 = conv3.reshape(num_hiddf,13)
    conv4 = conv4.reshape(num_hiddf)/(num_convf*xdim2*ydim2)
    conv5 = conv5.reshape(13)/(xdim2*ydim2*num_hiddf)

    for l in range(13):
        for j in range(num_hiddf):
            mul_w[:,:,j] = sess.run(conv3[j,l]*tf.tanh(np.sum(conv1[:,:,:]*conv2[:,:,:,j]+conv4[j],axis=2)))

        hvalue = np.mean(mul_w,axis=2)+conv5[l]
        print('k:',k,'l:',l)
        heat_mean[k,l,:,:] = hvalue


ds = xr.Dataset({'HeatValue': (('year','region', 'lat','lon'), heat_mean)},coords={'year': np.arange(1,37,1),'region': np.arange(1,14,1),'lon': np.arange(10.0,360.5,20.0), 'lat': np.arange(-45.0,61.0,20.0)})
ds.to_netcdf('HeatValue_'+str(monst).zfill(2) + "st_lead" + str(nstep + 1).zfill(2)+'_tarDJF.nc')



