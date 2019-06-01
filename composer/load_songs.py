import midi
import os
import util
import numpy as np
import sys, random, os

VERBOSE = False
HOMEBASE = sys.argv[1]

patterns = {}
dirs = ["Music", "download", "rag", "pop", "misc"]
all_samples = []
all_lens = []
print "Loading Songs..."
for dir in dirs:
	for root, subdirs, files in os.walk(HOMEBASE + dir):
		for file in files:
			path = root + os.sep + file
			if not (path.endswith('.mid') or path.endswith('.midi')):
				continue
			try:
				if VERBOSE:
					print "reading", path
				samples = midi.midi_to_samples(path)
			except:
				print "ERROR ", path
				continue
			if len(samples) < 8:
				continue

			if VERBOSE:
				print "Adding", path
			samples, lens = util.generate_add_centered_transpose(samples)
			all_samples += samples
			all_lens += lens

assert(sum(all_lens) == len(all_samples))
print "Saving " + str(len(all_samples)) + " samples..."
print "From " + str(len(all_lens)) + " songs..."
all_samples = np.array(all_samples, dtype=np.uint8)
all_lens = np.array(all_lens, dtype=np.uint32)
np.save(HOMEBASE + 'samples.npy', all_samples)
np.save(HOMEBASE + 'lengths.npy', all_lens)
print "Done"
