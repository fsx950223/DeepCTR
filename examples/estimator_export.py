import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from deepctr.estimator import DCNMixEstimator

if __name__ == "__main__":

    # 1.generate feature_column for linear part and dnn part

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    dnn_feature_columns = []
    linear_feature_columns = []

    for i, feat in enumerate(sparse_features):
        dnn_feature_columns.append(tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_identity(feat, 1000), 4))
        linear_feature_columns.append(tf.feature_column.categorical_column_with_identity(feat, 1000))
    for feat in dense_features:
        dnn_feature_columns.append(tf.feature_column.numeric_column(feat))
        linear_feature_columns.append(tf.feature_column.numeric_column(feat))

    # 3.Define Model,train,predict and evaluate
    model = DCNMixEstimator(linear_feature_columns, dnn_feature_columns, task='binary', model_dir='./test',
                            config=tf.estimator.RunConfig(tf_random_seed=2021))
    
    d = {k: tf.placeholder(tf.int64, shape=(None, 1), name=k) for k in sparse_features}
    d.update({k: tf.placeholder(tf.float32, shape=(None, 1), name=k) for k in dense_features})
    serving_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(d)  # Pass in a dummy input batch.
    model.export_saved_model('./saved_model', serving_input_fn)

