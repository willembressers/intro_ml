# -*- coding: utf-8 -*-

# python core
import math
from multiprocessing import Pool
from multiprocessing import cpu_count

def chunk(items:list, count:int):
	# loop over the list in n-sized chunks
	for index in range(0, len(items), count):

		# yield the current n-sized chunk to the calling function
		yield items[index: index + count]

def devide_per_processor(items:list) -> list:
    # determine the number of concurrent processes to launch when
	# distributing the load across the system, then create the list
	# of process IDs
    num_processors = cpu_count()
    print(f'num_processors:{num_processors}')

    # Devide the list over the number of processors
    num_per_processor = len(items) / float(num_processors)
    num_per_processor = int(math.ceil(num_per_processor))
    print(f'num_per_processor:{num_per_processor}')

    # chunk the image paths into N (approximately) equal sets, one
    # set of image paths for each individual process
    return list(chunk(items, num_per_processor))

def this(processor, items:list):
    num_processors = cpu_count()

    # devide the list over the processors
    chunks = devide_per_processor(items)

    # construct and launch the processing pool
    print("[INFO] launching pool using {} processes...".format(num_processors))
    pool = Pool(processes=num_processors)
    pool.map(processor, chunks)

    # close the pool and wait for all processes to finish
    print("[INFO] waiting for processes to finish...")
    pool.close()
    pool.join()
    print("[INFO] multiprocessing complete")