import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylab import rcParams
from src.util import createDataset \
    , transformDataset \
    , splitTrainTest \
    , defineModel \
    , defineAutoencoder \
    , predictAutoencoder \
    , getLatentFeaturesSimilariryDF \
    , getReconstructionErrorsDF \
    , portfolio_selection\
    , calcMarkowitzPortfolio\
    , getAverageReturnsDF\
    , chunks\
    , calc_delta_matrix\
    , generate_outlier_series\
    , convert_relative_changes_to_absolute_values

from sklearn import preprocessing
from keras.utils.vis_utils import  plot_model
from keras.models import Model
import copy


x = np.arange(0,4*np.pi,0.01)   # start,stop,step
y = np.sin(x) +10
z = np.cos(x) +10

y2 = np.sin(x) + 2*x +10
z2 = np.cos(x) + 2*x +10

y3 = np.sin(x) - 2*x +10
z3 = np.cos(x) - 2*x +10





plt.plot(x,z,x,z2)
plt.show()


plt.plot(x,y,x,z)
plt.show()

plt.plot(x,y2,x,z2)
plt.show()

plt.plot(x,y3,x,z3)
plt.show()



plt.plot(x,y2,x,z3)
plt.show()

series_y = generate_outlier_series(median=630, err=2, outlier_err=100, size=80, outlier_size=5)
plt.plot(range(len(series_y)),series_y)
plt.show()



c = np.cov(y2,z2)

cor = np.corrcoef(z,y2)



numpy_data = {'y':  y
            ,'z':  z
            , 'y2':  y2
            ,'z2':  z2}

df = pd.DataFrame(data=numpy_data)

df_corr = df.corr()


print('-' * 20 + 'Transform dataset')
# shift values up by 1 to avoid inf and nan
df_1 = df
df_pct_change = (df - df.shift(1) )/df.shift(1)  # df_1.pct_change(1).astype(float)
df_pct_change = df_pct_change.replace([np.inf, -np.inf], np.nan)
df_pct_change = df_pct_change.fillna(method='ffill')
# the percentage change function will make the first two rows equal to nan
df_pct_change = df_pct_change.tail(len(df_pct_change)-2)

print('-' * 20 + 'Step 1 : Returns vs. recreation error (recreation_error)')
print('-' * 25 + 'Transform dataset with MinMax Scaler')
df_scaler = preprocessing.MinMaxScaler()
df_pct_change_normalised = df_scaler.fit_transform(df_pct_change)

# define autoencoder
print('-' * 25 + 'Define autoencoder model')
num_stock = len(df_pct_change.columns)
autoencoder = defineAutoencoder(num_stock=num_stock, encoding_dim=5, verbose=0)
plot_model(autoencoder, to_file='img/model_autoencoder_1.png', show_shapes=True,
           show_layer_names=True)

# train autoencoder
print('-' * 25 + 'Train autoencoder model')
autoencoder.fit(df_pct_change_normalised, df_pct_change_normalised, shuffle=False, epochs=500, batch_size=50,
                verbose=0)

# predict autoencoder
print('-' * 25 + 'Predict autoencoder model')
reconstruct = autoencoder.predict(df_pct_change_normalised)

# Inverse transform dataset with MinMax Scaler
print('-' * 25 + 'Inverse transform dataset with MinMax Scaler')
reconstruct_real = df_scaler.inverse_transform(reconstruct)

print('-' * 25 + 'Calculate L2 norm as reconstruction loss metric')
df_recreation_error = getReconstructionErrorsDF(df_pct_change= df_pct_change
                                                , reconstructed_data=reconstruct_real)

# -------------------------------------------------------
#           Step2:
# -------------------------------------------------------


print('-' * 20 + 'Step 2 : Returns vs. latent feature similarity')
print('-' * 25 + 'Transpose dataset')
df_pct_change_transposed = df_pct_change.transpose()

print('-' * 25 + 'Transform dataset with MinMax Scaler')
df_scaler = preprocessing.MinMaxScaler()
df_pct_change_transposed_normalised = df_scaler.fit_transform(df_pct_change_transposed)

# define autoencoder
print('-' * 25 + 'Define autoencoder model')
num_stock = len(df_pct_change_transposed.columns)
autoencoderTransposed = defineAutoencoder(num_stock=num_stock, encoding_dim=5, verbose=0)

# train autoencoder
print('-' * 25 + 'Train autoencoder model')
autoencoderTransposed.fit(df_pct_change_transposed_normalised, df_pct_change_transposed_normalised,
                          shuffle=True, epochs=500, batch_size=50, verbose=0)

# Get the latent feature vector
print('-' * 25 + 'Get the latent feature vector')
autoencoderTransposedLatent = Model(inputs=autoencoderTransposed.input,
                                    outputs=autoencoderTransposed.get_layer('Encoder_Input').output)
plot_model(autoencoderTransposedLatent, to_file='img/model_autoencoder_2.png', show_shapes=True,
           show_layer_names=True)

# predict autoencoder model
print('-' * 25 + 'Predict autoencoder model')
latent_features = autoencoderTransposedLatent.predict(df_pct_change_transposed_normalised)

print('-' * 25 + 'Calculate L2 norm as similarity metric')
df_similarity = getLatentFeaturesSimilariryDF(df_pct_change=df_pct_change
                                              , latent_features=latent_features)



# --------------------------------------------- #
#       WORK IN PROGRESS
# --------------------------------------------- #

fontsize = 12
stable_stocks = False
unstable_stocks = True

plot_original_values = True
plot_delta_values = False
# filtering out values with low variance in the last x days
df_result_filtered1 = copy.deepcopy(df_pct_change)

'''
df_result_filtered1 = df_result_filtered1.drop(
    columns=df_result_filtered1.columns[((df_result_filtered1.tail(300).diff(1) == 0).mean() >= 0.8)],
    axis=1)
# df_result_filtered1 = df_result_filtered1.reset_index()

# df_result_filtered1 = df_result_filtered1.loc[:, df_result_filtered1.tail(100).var() != 0.0]

# df_result_filtered1 = df_result_filtered1.filter(like='_Close')[df_result.filter(like='_Close').tail(300) != 0]
res = df_result_filtered1.pct_change(1).mean() * 100
l_res_new = []
l_res = res.index

[l_res_new.append(c.split('_')[0]) for c in l_res]
df_recreation_error_filtered = df_recreation_error[df_recreation_error.index.isin(l_res_new)]
'''

df_recreation_error_filtered = df_recreation_error

df_stable_stocks = df_recreation_error_filtered.sort_values(by=['recreation_error'], ascending=True)
l_stable_stocks = np.array(df_stable_stocks.head(5).index)

df_unstable_stocks = df_recreation_error_filtered.sort_values(by=['recreation_error'], ascending=False)
l_unstable_stocks = np.array(df_unstable_stocks.head(5).index)

if stable_stocks:
    list = l_stable_stocks
    title = 'Original versus autoencoded stock price for low recreation error (stable stocks)'

if unstable_stocks:
    list = l_unstable_stocks
    title = 'Original versus autoencoded stock price for high recreation error (unstable stocks)'


plt.figure()
plt.rcParams["figure.figsize"] = (8, 14)
plt.title(title, y=1.08)
plt.box(False)
fig, ax = plt.subplots(len(list), 1)

i = 0
for stock in list:


    which_stock = df_pct_change.columns.get_loc(stock)
    which_stock_name = df_pct_change.columns[which_stock,]


    ## plot for comparison
    if plot_original_values:
        df_reconstruct_real = pd.DataFrame(data=reconstruct_real[:,which_stock])
        stock_autoencoder_1 = convert_relative_changes_to_absolute_values(
            relative_values=df_reconstruct_real, initial_value=df.iloc[1, which_stock]) # the initial value is the second one as the first one is nan because of the delta calculation

        print('Plotting original values')
        ax[i].plot(df.iloc[2:,which_stock])
        ax[i].plot(df.index[2:], stock_autoencoder_1[:])

    if plot_delta_values:
        print('Plotting delta values')
        ax[i].plot(df_pct_change.iloc[:, which_stock])
        ax[i].plot(df_pct_change.index[:], reconstruct_real[:, which_stock])

    ax[i].legend(['Original ' + str(which_stock_name), 'Autoencoded ' + str(which_stock_name)],
                 frameon=False)

    # set title
    # plt.set_title('Original stock price [{}] versus autoencoded stock price '.format(column), fontsize=fontsize)

    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
    ax[i].spines['left'].set_visible(False)
    ax[i].spines['bottom'].set_visible(False)

    ax[i].axes.get_xaxis().set_visible(False)

    i = i + 1

plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.show()









df_delta_mat = calc_delta_matrix(df_similarity['similarity_score'])