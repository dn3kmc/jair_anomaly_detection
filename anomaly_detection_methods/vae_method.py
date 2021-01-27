import pandas as pd
import time
import anomaly_detection_methods_helpers as ah
import matplotlib.pyplot as plt
import sys  
sys.path.append("../characteristics") 
import characteristics_helpers as ch
import numpy as np
import warnings  
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    warnings.filterwarnings("ignore",category=Warning)
    import tensorflow as tf
    from tensorflow import keras as K
    from donut import complete_timestamp, standardize_kpi
    from donut import Donut
    from tfsnippet.modules import Sequential
    from donut import DonutTrainer, DonutPredictor
    from donut import iterative_masked_reconstruct 

def my_func_float(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  return arg

def my_func_int(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.int32)
  return arg

def vae_donut(ts_obj, window_size, mcmc_iteration, latent_dim, gaussian_window_size, step_size, plot_reconstruction=False, plot_anomaly_score=False):
    # authors use window_size = 120
    # mcmc_iteration = 10

    # https://github.com/kratzert/finetune_alexnet_with_tensorflow/issues/8
    tf.reset_default_graph()

    start = time.time()

    # if there are missing time steps, we DO NOT fill them with NaNs because donut will replace them with 0s
    # using complete_timestamp
    # see line 6 in https://github.com/NetManAIOps/donut/blob/master/donut/preprocessing.py
    timestamp, values, labels = ts_obj.dataframe["timestamp"].values, ts_obj.dataframe["value"].values, np.zeros_like(ts_obj.dataframe["value"].values, dtype=np.int32)
    
    # print(len(timestamp))
    # print(len(values))
    # print(len(labels))

    # Complete the timestamp, and obtain the missing point indicators
    # replaces  missing with 0s.

    # donut cannot handle this date format for some reason
    if ts_obj.dateformat == "%Y-%m":
        rng = pd.date_range('2000-01-01', periods=len(values), freq='T')
        timestamp, missing, (values, labels) = complete_timestamp(rng, (values, labels))
    else:
        timestamp, missing, (values, labels) = complete_timestamp(timestamp, (values, labels))

    # print(len(timestamp))
    # print(len(values))
    # print(len(labels))
    # print(sum(missing))

    # Standardize the training and testing data.
    values, mean, std = standardize_kpi(values, excludes=np.logical_or(labels, missing))

    with tf.variable_scope('model') as model_vs:
        model = Donut(
            h_for_p_x=Sequential([
                K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                               activation=tf.nn.relu),
                K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                               activation=tf.nn.relu),
            ]),
            h_for_q_z=Sequential([
                K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                               activation=tf.nn.relu),
                K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                               activation=tf.nn.relu),
            ]),
            x_dims=window_size,
            z_dims=latent_dim,
        )

        trainer = DonutTrainer(model=model, model_vs=model_vs)
        predictor = DonutPredictor(model)

        with tf.Session().as_default():
            trainer.fit(values, labels, missing, mean, std)
            score = predictor.get_score(values, missing)

            # if time series is [1,2,3,4...] and ts_length is 3
            # this gives us [[1,2,3],[2,3,4]...]
            ts_strided = ah.as_sliding_window(values, window_size)
            ts_strided = my_func_float(np.array(ts_strided, dtype=np.float32))
            missing_strided = ah.as_sliding_window(missing, window_size)
            missing_strided = my_func_int(np.array(missing_strided, dtype=np.int32))

            # print(ts_strided)
            # print(missing_strided)

            x = model.vae.reconstruct(
                iterative_masked_reconstruct(
                    reconstruct=model.vae.reconstruct,
                    x=ts_strided,
                    mask=missing_strided,
                    iter_count=mcmc_iteration,
                    back_prop=False
                )
            )

            # `x` is a :class:`tfsnippet.stochastic.StochasticTensor`, from which
            # you may derive many useful outputs, for example:
            # print(x.tensor.eval())  # the `x` samples
            # print(x.log_prob(group_ndims=0).eval())  # element-wise log p(x|z) of sampled x
            # print(x.distribution.log_prob(ts_strided).eval())  # the reconstruction probability
            # print(x.distribution.mean.eval(), x.distribution.std.eval())  # mean and std of p(x|z)

            tensor_reconstruction_probabilities = x.distribution.log_prob(ts_strided).eval()

            # because of the way strided works, we use the first 120 anomaly scores in the first slide
            # and then for remaining slides, we use the last point/score
            reconstruction_probabilities = list(tensor_reconstruction_probabilities[0])
            for i in range(len(tensor_reconstruction_probabilities)):
                if i != 0 :
                    slide =  tensor_reconstruction_probabilities[i]
                    reconstruction_probabilities.append(slide[-1])

    # print(len(reconstruction_probabilities))
    # print(len(ts_obj.dataframe))

    if ts_obj.miss:
        ref_date_range = ch.get_ref_date_range(ts_obj.dataframe, ts_obj.dateformat, ts_obj.timestep)
        gaps = ref_date_range[~ref_date_range.isin(ts_obj.dataframe["timestamp"])]
        filled_df = ch.fill_df(ts_obj.dataframe, ts_obj.timestep, ref_date_range, "fill_nan")
        # print("NaNs exist?: ",filled_df['value'].isnull().values.any())
        filled_df["reconstruction_probabilities"] = reconstruction_probabilities
        # remove nans
        filled_df = filled_df.dropna()
        reconstruction_probabilities = list(filled_df["reconstruction_probabilities"].values)
        
    # print(len(reconstruction_probabilities))
    # print(len(ts_obj.dataframe))

    reconstruction_probabilities = [abs(item) for item in reconstruction_probabilities]

    anomaly_scores = anomaly_scores = ah.determine_anomaly_scores_error(reconstruction_probabilities, np.zeros_like(reconstruction_probabilities), ts_obj.get_length(), gaussian_window_size, step_size)


    end = time.time()

    if plot_reconstruction:
        plt.subplot(211)
        # see lines 98 to 100 of https://github.com/NetManAIOps/donut/blob/master/donut/prediction.py
        plt.title("Negative of Reconstruction Probabilities")
        plt.plot(reconstruction_probabilities)
        # plt.ylim([.99,1])
        plt.subplot(212)
        plt.title("Time Series")
        plt.plot(ts_obj.dataframe["value"].values)   
        plt.axvline(ts_obj.get_probationary_index(), color="black", label="probationary line")
        plt.tight_layout()
        plt.show()


    if plot_anomaly_score:
        plt.subplot(211)
        plt.title("Anomaly Scores")
        plt.plot(anomaly_scores)
        plt.ylim([.998,1])
        plt.subplot(212)
        plt.title("Time Series")
        plt.plot(ts_obj.dataframe["value"].values)   
        plt.axvline(ts_obj.get_probationary_index(), color="black", label="probationary line")
        plt.tight_layout()
        plt.show()

    return {"Anomaly Scores": anomaly_scores,
            "Time": end - start,
            "Reconstruction Probabilities": reconstruction_probabilities}
