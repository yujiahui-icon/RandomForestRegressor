import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sklearn
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from mhfp.encoder import MHFPEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
df = pd.read_csv('../data/raw/Exp_rf.csv')
x = df['ssid']
y = df['dGsolv']
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.1, random_state=42)


mh_x = []
for i in x_train:
    x_mhfp = MHFPEncoder.secfp_from_smiles(i, length=1024, radius=3, rings=True,kekulize=True,
                                       sanitize=False)
    mh_x.append(x_mhfp)
x = np.array(mh_x)

rfr = RandomForestRegressor()
rfr.fit(x,y_train)

mh_x_test = []
for i in x_test:
    x_test_mhfp = MHFPEncoder.secfp_from_smiles(i, length=1024, radius=3, rings=True,kekulize=True,
                                       sanitize=False)
    mh_x_test.append(x_test_mhfp)
x_test_1 = np.array(mh_x_test)


pred = rfr.predict(x_test_1)
# print(metrics.classification_report(pred,y_test))
print(pred)
# y_tests = []
# r2_scores = []
# rmse_scores = []


# r_squared = r2_score(test_file['labels'], preds_test)
# rmse = mean_squared_error(test_file['labels'], preds_test) ** 0.5
# mae = mean_absolute_error(test_file['labels'], preds_test)


# def make_plot(true, pred, rmse, r2_score, mae, name):
#     fontsize = 16
#     fig, ax = plt.subplots(figsize=(10, 8))
#     r2_patch = mpatches.Patch(label="R2 = {:.3f}".format(r2_score), color="#008080")
#     rmse_patch = mpatches.Patch(label="RMSE = {:.2f}".format(rmse), color="#008B8B")
#     mae_patch = mpatches.Patch(label="MAE = {:.2f}".format(mae), color="#20B2AA")
#     plt.xlim(-30, 10)
#     plt.ylim(-30, 10)
#     plt.tick_params(labelsize=20)
#     plt.scatter(true, pred, s=100, alpha=0.5, color="#008B8B",)
#     plt.plot(np.arange(-60, 10, 0.01), np.arange(-60, 10, 0.01), ls="--", c=".3")
#     plt.legend(handles=[r2_patch, rmse_patch, mae_patch], fontsize=18)
#     ax.set_xlabel('Experimental, ΔGsolv [kcal/mol]', fontsize=18)
#     ax.set_ylabel('Predicted, ΔGsolv [kcal/mol]', fontsize=18)
#     ax.set_title(name, fontsize=fontsize)
#     return fig
#
# print(f"  R2 {r_squared:.3f},RMSE {rmse:.2f},MAE {mae:.2f}")
#
# # fig = make_plot(true_test, preds_test, rmse, r_squared, mae, ' ')
# # fig.savefig('pictures/exp_t_scatter.tiff', dpi=600)
#
# def make_hist(trues, preds):
#     fontsize = 12
#     fig_hist, ax = plt.subplots(nrows=1, ncols=2,figsize=(8, 8))
#
#     plt.hist(trues, bins=30, label='true',
#                      facecolor='#1E90FF', histtype='bar', alpha=0.8)
#     plt.hist(preds, bins=30, label='predict',
#                      facecolor='#2E8B57', histtype='bar', alpha=0.6)
#
#     plt.xlabel('ΔGsolv [kcal/mol]', fontsize=18)
#     plt.ylabel('amount', fontsize=18)
#     plt.tick_params(labelsize=18)
#     plt.legend(loc='upper left', fontsize=18)
#     # ax.set_title(fontsize=fontsize)
#     return fig_hist
#
# fig_hist = make_hist(true_test, preds_test)
# fig_hist.savefig('pictures/exp_t_hist.tiff', dpi=600)

