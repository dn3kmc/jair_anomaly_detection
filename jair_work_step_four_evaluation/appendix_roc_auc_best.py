import sys  
sys.path.append("../time_series")  
sys.path.append("../anomaly_detection_methods")

import anomaly_detection_methods_helpers as ah
from time_series import TimeSeries
from os import listdir
from os.path import isfile, join
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import math
import matplotlib.pyplot as plt

count = 0
group = 0

fig = plt.figure(figsize=(20,10))


# https://stackoverflow.com/questions/28719067/roc-curve-and-cut-off-point-python
def cutoff_youdens_j(fpr,tpr,thresholds):
    j_scores = tpr-fpr
    j_ordered = sorted(zip(j_scores,thresholds))
    return j_ordered[-1][1]

mypath = "../jair_work_step_one_determine_characteristics/"
for f in listdir(mypath):
	if "ts_object" in f:
		ts = joblib.load(mypath + f)

		# print(ts.name)
		methods_applicable = []

		threshold_dict = {}

		# anomalous
		anomalous_roc_auc_best_path = "anomalous_roc_auc_best/"
		for an in listdir(anomalous_roc_auc_best_path):
			if ts.name in an:
				methods_applicable.append("Anomalous")
				an_best_roc_auc_dict = joblib.load(anomalous_roc_auc_best_path + an)
				fpr_an = an_best_roc_auc_dict["FPRS"]
				tpr_an = an_best_roc_auc_dict["TPRS"]
				roc_auc_an = an_best_roc_auc_dict["AUC"]
				threshold_dict["Anomalous"] = min(cutoff_youdens_j(fpr_an,tpr_an,an_best_roc_auc_dict["Thresholds"]),1)
				# print("Best Threshold for Anomalous: ", cutoff_youdens_j(fpr_an,tpr_an,an_best_roc_auc_dict["Thresholds"]))


		# prophet
		prophet_roc_auc_best_path = "prophet_roc_auc_best/"
		for prophet in listdir(prophet_roc_auc_best_path):
			if ts.name in prophet:
				methods_applicable.append("Prophet")
				prophet_best_roc_auc_dict = joblib.load(prophet_roc_auc_best_path + prophet)
				fpr_prophet= prophet_best_roc_auc_dict["FPRS"]
				tpr_prophet = prophet_best_roc_auc_dict["TPRS"]
				roc_auc_prophet = prophet_best_roc_auc_dict["AUC"]
				threshold_dict["Prophet"] = min(cutoff_youdens_j(fpr_prophet,tpr_prophet,prophet_best_roc_auc_dict["Thresholds"]),1)
				# print("Best Threshold for Prophet: ", cutoff_youdens_j(fpr_prophet,tpr_prophet,prophet_best_roc_auc_dict["Thresholds"]))

		# glim
		glim_roc_auc_best_path = "glim_roc_auc_best/"
		for glim in listdir(glim_roc_auc_best_path):
			if ts.name in glim:
				methods_applicable.append("GLiM")
				glim_best_roc_auc_dict = joblib.load(glim_roc_auc_best_path + glim)
				fpr_glim = glim_best_roc_auc_dict["FPRS"]
				tpr_glim = glim_best_roc_auc_dict["TPRS"]
				roc_auc_glim = glim_best_roc_auc_dict["AUC"]
				threshold_dict["GLiM"] = min(cutoff_youdens_j(fpr_glim,tpr_glim,glim_best_roc_auc_dict["Thresholds"]),1)
				# print("Best Threshold for GLiM: ", cutoff_youdens_j(fpr_glim,tpr_glim,glim_best_roc_auc_dict["Thresholds"]))

		# matrix profile
		matrix_profile_roc_auc_best_path = "matrix_profile_roc_auc_best/"
		for mp in listdir(matrix_profile_roc_auc_best_path):
			if ts.name in mp:
				methods_applicable.append("Matrix Profile")
				mp_best_roc_auc_dict = joblib.load(matrix_profile_roc_auc_best_path + mp)
				fpr_mp = mp_best_roc_auc_dict["FPRS"]
				tpr_mp = mp_best_roc_auc_dict["TPRS"]
				roc_auc_mp = mp_best_roc_auc_dict["AUC"]
				threshold_dict["Matrix Profile"] = min(cutoff_youdens_j(fpr_mp,tpr_mp,mp_best_roc_auc_dict["Thresholds"]),1)
				# print("Best Threshold for Matrix Profile: ", cutoff_youdens_j(fpr_mp,tpr_mp,mp_best_roc_auc_dict["Thresholds"]))

		# pbad
		pbad_roc_auc_best_path = "pbad_roc_auc_best/"
		for pbad in listdir(pbad_roc_auc_best_path):
			if ts.name in pbad:
				methods_applicable.append("PBAD")
				pbad_best_roc_auc_dict = joblib.load(pbad_roc_auc_best_path + pbad)
				fpr_pbad = pbad_best_roc_auc_dict["FPRS"]
				tpr_pbad = pbad_best_roc_auc_dict["TPRS"]
				roc_auc_pbad = pbad_best_roc_auc_dict["AUC"]
				threshold_dict["PBAD"] = min(cutoff_youdens_j(fpr_pbad,tpr_pbad,pbad_best_roc_auc_dict["Thresholds"]),1)
				# print("Best Threshold for PBAD: ", cutoff_youdens_j(fpr_pbad,tpr_pbad,pbad_best_roc_auc_dict["Thresholds"]))

		# sarimax
		sarimax_roc_auc_best_path = "sarimax_roc_auc_best/"
		for sarimax in listdir(sarimax_roc_auc_best_path):
			if ts.name in sarimax:
				methods_applicable.append("SARIMAX")
				sarimax_best_roc_auc_dict = joblib.load(sarimax_roc_auc_best_path + sarimax)
				fpr_sarimax = sarimax_best_roc_auc_dict["FPRS"]
				tpr_sarimax = sarimax_best_roc_auc_dict["TPRS"]
				roc_auc_sarimax = sarimax_best_roc_auc_dict["AUC"]
				threshold_dict["SARIMAX"] = min(cutoff_youdens_j(fpr_sarimax,tpr_sarimax,sarimax_best_roc_auc_dict["Thresholds"]),1)
				# print("Best Threshold for SARIMAX: ", cutoff_youdens_j(fpr_sarimax,tpr_sarimax,sarimax_best_roc_auc_dict["Thresholds"]))

		# stl
		stl_roc_auc_best_path = "stl_roc_auc_best/"
		for stl in listdir(stl_roc_auc_best_path):
			if ts.name in stl:
				methods_applicable.append("STL")
				stl_best_roc_auc_dict = joblib.load(stl_roc_auc_best_path + stl)
				fpr_stl = stl_best_roc_auc_dict["FPRS"]
				tpr_stl = stl_best_roc_auc_dict["TPRS"]
				roc_auc_stl = stl_best_roc_auc_dict["AUC"]
				threshold_dict["STL"] = min(cutoff_youdens_j(fpr_stl,tpr_stl,stl_best_roc_auc_dict["Thresholds"]),1)
				# print("Best Threshold for STL: ", cutoff_youdens_j(fpr_stl,tpr_stl,stl_best_roc_auc_dict["Thresholds"]))

		# streaming hs
		hstree_roc_auc_best_path = "hstree_roc_auc_best/"
		for hstree in listdir(hstree_roc_auc_best_path):
			if ts.name in hstree:
				methods_applicable.append("HS Tree")
				hstree_best_roc_auc_dict = joblib.load(hstree_roc_auc_best_path + hstree)
				fpr_hstree = hstree_best_roc_auc_dict["FPRS"]
				tpr_hstree = hstree_best_roc_auc_dict["TPRS"]
				roc_auc_hstree = hstree_best_roc_auc_dict["AUC"]
				threshold_dict["HS Tree"] = min(cutoff_youdens_j(fpr_hstree,tpr_hstree,hstree_best_roc_auc_dict["Thresholds"]),1)
				# print("Best Threshold for HS Tree: ", cutoff_youdens_j(fpr_hstree,tpr_hstree,hstree_best_roc_auc_dict["Thresholds"]))

		# twitter
		twitter_roc_auc_best_path = "twitter_roc_auc_best/"
		for twitter in listdir(twitter_roc_auc_best_path):
			if ts.name in twitter:
				methods_applicable.append("Twitter")
				twitter_best_roc_auc_dict = joblib.load(twitter_roc_auc_best_path + twitter)
				fpr_twitter = twitter_best_roc_auc_dict["FPRS"]
				tpr_twitter = twitter_best_roc_auc_dict["TPRS"]
				roc_auc_twitter = twitter_best_roc_auc_dict["AUC"]
				threshold_dict["Twitter"] = min(cutoff_youdens_j(fpr_twitter,tpr_twitter,twitter_best_roc_auc_dict["Thresholds"]),1)
				# print("Best Threshold for Twitter: ", cutoff_youdens_j(fpr_twitter,tpr_twitter,twitter_best_roc_auc_dict["Thresholds"]))

		# vae
		vae_roc_auc_best_path = "vae_roc_auc_best/"
		for vae in listdir(vae_roc_auc_best_path):
			if ts.name in vae:
				methods_applicable.append("VAE")
				vae_best_roc_auc_dict = joblib.load(vae_roc_auc_best_path + vae)
				fpr_vae = vae_best_roc_auc_dict["FPRS"]
				tpr_vae = vae_best_roc_auc_dict["TPRS"]
				roc_auc_vae = vae_best_roc_auc_dict["AUC"]
				threshold_dict["VAE"] = min(cutoff_youdens_j(fpr_vae,tpr_vae,vae_best_roc_auc_dict["Thresholds"]),1)
				# print("Best Threshold for VAE: ", cutoff_youdens_j(fpr_vae,tpr_vae,vae_best_roc_auc_dict["Thresholds"]))

		# windowed gaussian
		windowed_gaussian_roc_auc_best_path = "windowed_gaussian_roc_auc_best/"
		for windowed_gaussian in listdir(windowed_gaussian_roc_auc_best_path):
			if ts.name in windowed_gaussian:
				methods_applicable.append("Windowed Gaussian")
				windowed_gaussian_best_roc_auc_dict = joblib.load(windowed_gaussian_roc_auc_best_path + windowed_gaussian)
				fpr_windowed_gaussian = windowed_gaussian_best_roc_auc_dict["FPRS"]
				tpr_windowed_gaussian = windowed_gaussian_best_roc_auc_dict["TPRS"]
				roc_auc_windowed_gaussian = windowed_gaussian_best_roc_auc_dict["AUC"]
				threshold_dict["Windowed Gaussian"] = min(cutoff_youdens_j(fpr_windowed_gaussian,tpr_windowed_gaussian,windowed_gaussian_best_roc_auc_dict["Thresholds"]),1)
				# print("Best Threshold for Windowed Gaussian: ", cutoff_youdens_j(fpr_windowed_gaussian,tpr_windowed_gaussian,windowed_gaussian_best_roc_auc_dict["Thresholds"]))

		# htm
		htm_roc_auc_best_path = "htm_roc_auc_best/"
		for htm in listdir(htm_roc_auc_best_path):
			if ts.name in htm:
				methods_applicable.append("HTM")
				htm_best_roc_auc_dict = joblib.load(htm_roc_auc_best_path + htm)
				fpr_htm = htm_best_roc_auc_dict["FPRS"]
				tpr_htm = htm_best_roc_auc_dict["TPRS"]
				roc_auc_htm = htm_best_roc_auc_dict["AUC"]
				threshold_dict["HTM"] = min(cutoff_youdens_j(fpr_htm,tpr_htm,htm_best_roc_auc_dict["Thresholds"]),1)
				# print("Best Threshold for HTM: ", cutoff_youdens_j(fpr_htm,tpr_htm,htm_best_roc_auc_dict["Thresholds"]))

		joblib.dump(threshold_dict, "dataset_youden_thresholds/" + ts.name + "_youden_thresholds")

		### plot ###

		lw = 2

		if count % 4 == 0:

			ax = fig.add_subplot(2,2,1)
			last = 1

		elif count % 4 == 1:

			ax = fig.add_subplot(2,2,2)
			last = 2

		elif count % 4 == 2:

			ax = fig.add_subplot(2,2,3)
			last = 3

		elif count % 4 == 3:

			ax = fig.add_subplot(2,2,4)
			last = 4

		else:

			print("ERROR")


		if "Anomalous" in methods_applicable:
			ax.plot(fpr_an, tpr_an,lw=lw, label='Anomalous ROC (AUC = %0.2f)' % roc_auc_an)

		if "Prophet" in methods_applicable:
			ax.plot(fpr_prophet, tpr_prophet,lw=lw, label='Prophet ROC (AUC = %0.2f)' % roc_auc_prophet)

		if "GLiM" in methods_applicable:
			ax.plot(fpr_glim, tpr_glim,lw=lw, label='GLiM ROC (AUC = %0.2f)' % roc_auc_glim)

		if "Matrix Profile" in methods_applicable:
			ax.plot(fpr_mp, tpr_mp,lw=lw, label='Matrix Profile ROC (AUC = %0.2f)' % roc_auc_mp,)

		if "PBAD" in methods_applicable:
			ax.plot(fpr_pbad, tpr_pbad,lw=lw, label='PBAD ROC (AUC = %0.2f)' % roc_auc_pbad,)

		if "SARIMAX" in methods_applicable:
			ax.plot(fpr_sarimax, tpr_sarimax,lw=lw, label='SARIMAX ROC (AUC = %0.2f)' % roc_auc_sarimax,)

		if "STL" in methods_applicable:
			ax.plot(fpr_stl, tpr_stl,lw=lw, label='STL ROC (AUC = %0.2f)' % roc_auc_stl,)

		if "HS Tree" in methods_applicable:
			ax.plot(fpr_hstree, tpr_hstree,lw=lw, label='HS Tree ROC (AUC = %0.2f)' % roc_auc_hstree,)

		if "Twitter" in methods_applicable:
			ax.plot(fpr_twitter, tpr_twitter,lw=lw, label='Twitter ROC (AUC = %0.2f)' % roc_auc_twitter,)

		if "VAE" in methods_applicable:
			ax.plot(fpr_vae, tpr_vae,lw=lw, label='VAE ROC (AUC = %0.2f)' % roc_auc_vae,)

		if "Windowed Gaussian" in methods_applicable:
			ax.plot(fpr_windowed_gaussian, tpr_windowed_gaussian,lw=lw, label='Windowed Gaussian ROC (AUC = %0.2f)' % roc_auc_windowed_gaussian,)

		if "HTM" in methods_applicable:
			ax.plot(fpr_htm, tpr_htm,lw=lw, label='HTM ROC (AUC = %0.2f)' % roc_auc_htm,)


		


		ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
		if last in [2,4]:
			ax.set_yticklabels([])
		# ax.set_xticklabels([])
		# ax.set_xlim([0.0, 1.0])
		# ax.set_ylim([0.0, 1.05])
		# ax.set_xlabel('FPR')
		# ax.set_ylabel('TPR')
		# print("\n\n\n\n\n\n", ts.name)
		ax.set_title("ROCs for " + ts.name[:-4])
		ax.legend(loc="lower right", prop={'size': 8, 'weight':'bold'})
		# plt.tight_layout()

		# at the fourth picture, new plot
		print(count)
		if count % 4 == 3:

			fig.tight_layout()

			plt.gcf().subplots_adjust(bottom=0.05)
			plt.gcf().subplots_adjust(left=0.02)

			fig.savefig("appendix_ROCS_group_" + str(group) + ".eps", format="eps")
			# plt.show()

			fig = plt.figure(figsize=(20,10))
			group += 1

		# 
		else:
			if count == 33:

				fig.tight_layout()

				plt.gcf().subplots_adjust(bottom=0.05)
				plt.gcf().subplots_adjust(left=0.02)

				fig.savefig("appendix_ROCS_group_"+str(group)+".eps", format="eps")

		count += 1

# bc we start from 0
print(count+1)
