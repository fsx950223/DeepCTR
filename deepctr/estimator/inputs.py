import tensorflow as tf


def input_fn_pandas(df, features, label=None, batch_size=256, num_epochs=1, shuffle=False, queue_capacity_factor=10,
                    num_threads=1):
    if label is not None:
        y = df[label]
    else:
        y = None

    return tf.compat.v1.estimator.inputs.pandas_input_fn(df[features], y, batch_size=batch_size, num_epochs=num_epochs,
                                               shuffle=shuffle, queue_capacity=batch_size * queue_capacity_factor,
                                               num_threads=num_threads)


def input_fn_tfrecord(filenames, feature_description, label=None, batch_size=256, num_epochs=1, num_parallel_calls=8,
                      shuffle_factor=10, prefetch_factor=1,
                      ):
    def _parse_examples(serial_exmp):
        try:
            features = tf.parse_single_example(serial_exmp, features=feature_description)
        except AttributeError:
            features = tf.io.parse_single_example(serial_exmp, features=feature_description)
        if label is not None:
            labels = features.pop(label)
            return features, labels
        return features

    def input_fn(mode, input_context=None):
        dataset = tf.data.TFRecordDataset(filenames)
        if input_context:
            dataset = dataset.shard(input_context.num_input_pipelines,
                                    input_context.input_pipeline_id)
        dataset = dataset.map(_parse_examples, num_parallel_calls=num_parallel_calls)
        if shuffle_factor > 0:
            dataset = dataset.shuffle(buffer_size=batch_size * shuffle_factor, reshuffle_each_iteration=False)

        dataset = dataset.repeat().batch(batch_size, drop_remainder=True)
        
        if prefetch_factor > 0:
            dataset = dataset.prefetch(buffer_size=batch_size * prefetch_factor)

        return dataset

    return input_fn
