import os
import time
import tensorflow as tf
tf.debugging.enable_traceback_filtering()


if __name__ == "__main__":
    cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
    if cluster_resolver.task_type in ("worker", "ps"):
        os.environ["GRPC_FAIL_FAST"] = "use_caller"
        server = tf.distribute.Server(
            cluster_resolver.cluster_spec(),
            job_name=cluster_resolver.task_type,
            task_index=cluster_resolver.task_id,
            protocol="grpc",
            start=True)
        server.join()
        exit(0)

    from deepctr.models import FwFM, DCNMix
    from deepctr.feature_column import SparseFeat, DenseFeat

    from deepctr.estimator.inputs import input_fn_tfrecord

    from virtual_data_generator import gen_data_df
    from utils import time_callback, pd_to_tfrecord
    from config import parse_args       
    args = parse_args()
    ###os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5' # 指定该代码文件的可见GPU为第一个和第二个
    # os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices # 指定该代码文件的可见GPU为第一个和第二个
    # os.environ['HIP_VISIBLE_DEVICES'] = cuda_visible_devices # 指定该代码文件的可见GPU为第一个和第二个
    # gpus=tf.config.list_physical_devices('GPU')
    '''
    gpus=tf.config.list_physical_devices('GPU')
    print('*'*20, 'worker:', gpus)#查看有多少个可用的GPU
    visible_devices = tf.config.get_visible_devices()
    print('*'*20, 'worker visible devices:', visible_devices)#查看有多少个可用的GPU
    '''
    ###worker_config.set_visible_devices(gpus[2:],'GPU')
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    dnn_feature_columns = []
    linear_feature_columns = []

    '''
    for i, feat in enumerate(sparse_features):
        dnn_feature_columns.append(tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_identity(feat, 1000), 128))
        linear_feature_columns.append(tf.feature_column.categorical_column_with_identity(feat, 1000))
    '''

    # 2.generate input data for model
    feature_description = {k: tf.io.FixedLenFeature(dtype=tf.int64, shape=1) for k in sparse_features}
    feature_description.update(
        {k: tf.io.FixedLenFeature(dtype=tf.int64, shape=1) for k in dense_features})
    feature_description['label'] = tf.io.FixedLenFeature(dtype=tf.int64, shape=1)

    # data = gen_data_df(args)
    # pd_to_tfrecord(data)
    # print(data)
    '''
    train_model_input = input_fn_tfrecord('./Criteo_virtual_TFR.tfrecords', feature_description, 'label', batch_size=256,
                                        num_epochs=100, shuffle_factor=10)
    '''

    train_dataset = input_fn_tfrecord('./Criteo_virtual_TFR.tfrecords', feature_description, 'label', batch_size=args.batch_size,
                                        num_epochs=args.epochs, shuffle_factor=10)(None)
    test_dataset = input_fn_tfrecord('./Criteo_virtual_TFR.tfrecords', feature_description, 'label',
                                        batch_size=8, num_epochs=1, shuffle_factor=0)(None)
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=args.embedding_hash_size, embedding_dim=args.embedding_dims, use_hash=True, dtype=tf.int64) for feat in sparse_features] + [DenseFeat(feat, 1) for feat in dense_features]
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns
    
    # 3.Define Model,train,predict and evaluate
    # variable_partitioner=(
    #     tf.distribute.experimental.partitioners.MinSizePartitioner(
    #         min_shard_bytes=(256 << 6),
    #         max_shards = 2))
    variable_partitioner = None
    strategy = tf.distribute.ParameterServerStrategy(cluster_resolver, variable_partitioner=variable_partitioner)
    '''
    model = DCNMixEstimator(linear_feature_columns, dnn_feature_columns, task='binary',
                            config=tf.estimator.RunConfig(tf_random_seed=2021, train_distribute=strategy))
    '''
    with strategy.scope():
        if(args.model_type == "dcnv2"):
            model = DCNMix(linear_feature_columns, dnn_feature_columns, task='binary')
        elif(args.model_type == "fwfm"):
            model = FwFM(linear_feature_columns, dnn_feature_columns, task='binary')
        else:
            raise NotImplementedError("Unknown model")

        ##### Modified by Ngaiman Chow on 2023-3-28 for we need to statistics the running time of training
        '''
        tf.estimator.train_and_evaluate(
            model,
            train_spec=tf.estimator.TrainSpec(input_fn=train_model_input),
            eval_spec=tf.estimator.EvalSpec(input_fn=test_model_input)
        )
        '''
        model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'])
    # input_options = tf.distribute.InputOptions()
    # train_dataset = tf.keras.utils.experimental.DatasetCreator(train_model_input, input_options=input_options)
    # test_dataset = tf.keras.utils.experimental.DatasetCreator(test_model_input, input_options=input_options)
    class PerformanceTime(tf.keras.callbacks.Callback):
        def __init__(self):
            self.total_times=[]
            self.start_time=0
        
        def on_epoch_begin(self, epoch, logs=None):
            self.start_time = time.perf_counter()
            return super().on_epoch_begin(epoch, logs)

        def on_epoch_end(self, epoch, logs=None):
            self.total_times.append(time.perf_counter() - self.start_time)
            return super().on_epoch_end(epoch, logs)
        
    perf_time = PerformanceTime()
    model.fit(train_dataset, epochs=args.epochs, validation_data=test_dataset, steps_per_epoch=args.num_batch, validation_steps=0, callbacks=[perf_time])
    throughout = args.batch_size * args.num_batch * (args.epochs - 5) / sum(perf_time.total_times[5: ])
    print('###### throughout = %.2f (examples/s)'%(throughout))
    exit(0)


    