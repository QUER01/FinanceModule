import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylab import rcParams
from FinanceModule.util import createDataset \
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
    , chunks

from sklearn import preprocessing
from keras.utils.vis_utils import  plot_model
from keras.models import Model

x = np.arange(0,4*np.pi,0.1)   # start,stop,step
y = np.sin(x)
z = np.cos(x)

y2 = np.sin(x) + 2*x
z2 = np.cos(x) + 2*x

y3 = np.sin(x) - 2*x
z3 = np.cos(x) - 2*x

plt.plot(x,y3,x,z2)
plt.show()


plt.plot(x,y2,x,z3)
plt.show()



c = np.cov(y2,z2)

cor = np.correlate(z2,y2)



numpy_data = {'y':  y
            ,'z':  z
            , 'y2':  y2
            ,'z2':  z2
            , 'y3':  y3
            ,'z3':  z3}

df = pd.DataFrame(data=numpy_data)

print('-' * 20 + 'Transform dataset')
df_pct_change = df.pct_change(1).astype(float)
df_pct_change = df_pct_change.replace([np.inf, -np.inf], np.nan)
df_pct_change = df_pct_change.fillna(method='ffill')
# the percentage change function will make the firstrow equal to nan
df_pct_change = df_pct_change.tail(len(df_pct_change) - 2)

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


