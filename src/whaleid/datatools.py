"""Contains a class to generate a datapipeline"""


import tensorflow as tf
import numpy as np
import os


class datapipeline():
    """Class for a datapipeline based on tensorflow's dataset API"""
    def __init__(self, datadir, labels, batchsize):
        self.datadir = datadir
        self.labels = labels
        self.batchsize = batchsize


    def parse_function(self,files):
        """Function to map filenames to images"""
        # First image
        image_string = tf.read_file(files[0])
        image1 = tf.image.decode_jpeg(image_string,channels=3)
        # Convert to float values in [0, 1]
        image1 = tf.image.convert_image_dtype(image1, tf.float32)
        image1 = tf.image.resize_image_with_pad(image1, 256, 512)

        # Second image
        image_string = tf.read_file(files[1])
        image2 = tf.image.decode_jpeg(image_string,channels=3)
        # Convert to float values in [0, 1]
        image2 = tf.image.convert_image_dtype(image2, tf.float32)
        image2 = tf.image.resize_image_with_pad(image2, 256, 512)

        target_diff = [tf.string_to_number(files[2], out_type=tf.float32)]

        return image1, image2, target_diff


    def generator_fct(self):
        """Generator function. Creates filenames of images.
        With 50% probability different images of same class,
        with 50% probability images of different classes.
        """
        same = np.random.randint(2)
        if same == 1:    # same whale
            f1 = np.random.choice(self.labels[self.labels['Id_c'].str.startswith('new')==False]['Id_c'])
            files = np.random.choice(self.labels[self.labels['Id_c']==f1]['Image'], 2, replace=False)
            files = np.append(files,'0')
        else:
            row1 = np.random.randint(len(self.labels))
            id1 = self.labels['Id_c'][row1]
            file1 = self.labels['Image'][row1]
            file2 = np.random.choice(self.labels[self.labels['Id_c']!=id1]['Image'])
            files = np.array([file1,file2,'1'],dtype=object)
        files[0] = os.path.join(self.datadir,files[0])
        files[1] = os.path.join(self.datadir,files[1])
        yield files


    def build(self):
        """Build datapipeline"""
        dataset = tf.data.Dataset.from_generator(
                self.generator_fct,
                output_types= (tf.string),
                output_shapes=(tf.TensorShape([3])))
        dataset = dataset.map(self.parse_function, num_parallel_calls=1)
        dataset = dataset.prefetch(1)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.batchsize)

        return dataset
