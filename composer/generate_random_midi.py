#
#	Usage: generate_random_midi.py HOMEDIRECTORY NUMBER_OF_TRACKS
#
import sys, random, os
import midi
from keras.models import Model, Sequential, load_model
from keras import backend as K
from matplotlib import pyplot as plt
import numpy as np

HOMEBASE = sys.argv[1]
NUM_RAND_SONGS = int(sys.argv[2])
WRITEDIR = HOMEBASE + 'Out/'
MODELDIR = HOMEBASE + 'Out/History/'
MIDIDIR = HOMEBASE + 'RandomMidi/'

NUM_EPOCHS = 2000
LR = 0.001
CONTINUE_TRAIN = False
PLAY_ONLY = False
USE_EMBEDDING = False
USE_VAE = False
WRITE_HISTORY = True
DO_RATE = 0.1
BN_M = 0.9
VAE_B1 = 0.02
VAE_B2 = 0.1

BATCH_SIZE = 350
MAX_LENGTH = 16
PARAM_SIZE = 120
NUM_OFFSETS = 16 if USE_EMBEDDING else 1


print "Loading Model..."
model = load_model(MODELDIR + 'model.h5')
enc = Model(inputs=model.input, outputs=model.get_layer('pre_encoder').output)

func = K.function([model.get_layer('encoder').input, K.learning_phase()],
				  [model.layers[-1].output])
enc = Model(inputs=model.input, outputs=model.get_layer('pre_encoder').output)

rand_vecs = np.random.normal(0.0, 1.0, (NUM_RAND_SONGS, PARAM_SIZE))
print rand_vecs
np.save('rand.npy', rand_vecs)



def make_rand_songs(write_dir, rand_vecs):
	for i in xrange(rand_vecs.shape[0]):
		x_rand = rand_vecs[i:i+1]
		y_song = func([x_rand, 0])[0]
		midi.samples_to_midi(y_song[0], write_dir + 'rand' + str(i) + '.mid', ticks_per_sample = 0.125, thresh=0.20)


def make_rand_songs_normalized(write_dir, rand_vecs):
	if USE_EMBEDDING:
		x_enc = np.squeeze(enc.predict(x_orig))
	else:
		x_enc = np.squeeze(enc.predict(y_orig))

	x_mean = np.mean(x_enc, axis=0)
	x_stds = np.std(x_enc, axis=0)
	x_cov = np.cov((x_enc - x_mean).T)
	u, s, v = np.linalg.svd(x_cov)
	e = np.sqrt(s)

	print "Means: ", x_mean[:6]
	print "Evals: ", e[:6]

	np.save(write_dir + 'means.npy', x_mean)
	np.save(write_dir + 'stds.npy', x_stds)
	np.save(write_dir + 'evals.npy', e)
	np.save(write_dir + 'evecs.npy', v)

	x_vecs = x_mean + np.dot(rand_vecs * e, v)

	#fix to randomize the vector
	for i in range(len(x_vecs)):
		for k in range(len(x_vecs[0])):
			x_vecs[i][k]=np.random.random()*10-5

	print "Creating random Songs"
	make_rand_songs(write_dir, x_vecs)

	title = ''
	if '/' in write_dir:
		title = 'Epoch: ' + write_dir.split('/')[-2][1:]

	plt.clf()
	e[::-1].sort()
	plt.title(title)
	plt.bar(np.arange(e.shape[0]), e, align='center')
	plt.draw()
	plt.savefig(write_dir + 'evals.png')

	plt.clf()
	plt.title(title)
	plt.bar(np.arange(e.shape[0]), x_mean, align='center')
	plt.draw()
	plt.savefig(write_dir + 'means.png')

	plt.clf()
	plt.title(title)
	plt.bar(np.arange(e.shape[0]), x_stds, align='center')
	plt.draw()
	plt.savefig(write_dir + 'stds.png')



print "Loading Data..."
y_samples = np.load(HOMEBASE + 'samples.npy')
y_lengths = np.load(HOMEBASE + 'lengths.npy')
num_samples = y_samples.shape[0]
num_songs = y_lengths.shape[0]
print "Loaded " + str(num_samples) + " samples from " + str(num_songs) + " songs."
assert(np.sum(y_lengths) == num_samples)

print "Padding Songs..."
x_shape = (num_songs * NUM_OFFSETS, 1)
y_shape = (num_songs * NUM_OFFSETS, MAX_LENGTH) + y_samples.shape[1:]
x_orig = np.expand_dims(np.arange(x_shape[0]), axis=-1)
y_orig = np.zeros(y_shape, dtype=y_samples.dtype)

x_train = np.copy(x_orig)
y_train = np.copy(y_orig)

test_ix = 0
y_test_song = np.copy(y_train[test_ix:test_ix+1])


if not os.path.exists(MIDIDIR):
	os.makedirs(MIDIDIR)
make_rand_songs_normalized(MIDIDIR, rand_vecs)
