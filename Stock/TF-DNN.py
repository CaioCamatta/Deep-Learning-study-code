import numpy as np
from talib.abstract import *
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Select only 5-min ticks
stock_prices = pd.read_csv(r'E:\Code\Anaconda\Learning\Stock\all_stocks_5yr.csv')
print(stock_prices.shape)
#data = data.iloc[::5,:]
#print(data.shape)
print(stock_prices.head())


# BATCH CONTROL
class BatchHelper():
    def __init__(self, input_data):
        self.data = input_data
        self.training_data = pd.DataFrame()
        self.test_data = pd.DataFrame()

    def prepare_data(self):
        # Split dataset into separate datasets, one for each company
        grouped_stocks = list(self.data.groupby('Name'))

        # Dataframe that will store the data after it's been scaled.
        scaled_data = pd.DataFrame()
        scaled_training_data = pd.DataFrame()
        scaled_test_data = pd.DataFrame()

        print(f'Company index:', end=" ")
        for company in range(len(self.data['Name'].unique())):
            print(f'{company}', end=" ")
            # Get company data
            company_data = grouped_stocks[company][1]

            # Ignore company if the group contains missing data
            if(company_data.isnull().any(axis=1).any(axis=0)):
                continue

            # Normalize currency values ('open', 'high', 'low', 'close')
            sc_value = MinMaxScaler()
            sc_value.fit(np.array(company_data[['open', 'high', 'low', 'close']]).reshape(company_data.shape[0]*4).reshape(-1,1))
            company_data[['open', 'high', 'low', 'close']] = sc_value.transform(company_data[['open', 'high', 'low', 'close']])

            # Normalize volume
            sc_volume = MinMaxScaler()
            company_data[['volume']] = sc_volume.fit_transform(np.array(company_data[['volume']]))

            # Add data to final dataframe of scaled datasets
            scaled_data = scaled_data.append(company_data)
            if company < 420:
                scaled_training_data = scaled_training_data.append(company_data)
            else:
                scaled_test_data = scaled_test_data.append(company_data)


        print("Done preparing Data.")

        # Save normalized data
        self.data = scaled_data
        self.training_data = scaled_training_data
        self.test_data = scaled_test_data

        # Save grouped stocks to improve next_batch times
        grouped_training_stocks = list(self.training_data.groupby('Name'))
        grouped_test_stocks = list(self.test_data.groupby('Name'))
        self.grouped_training_stocks = grouped_training_stocks
        self.grouped_test_stocks = grouped_test_stocks

        # Return train (420 companies) and test (about 70 companies)
        return self.training_data, self.test_data

    def get_indicators(self, input_data):
        """ Calculates indicators for the passed data. Data must be a regularized np array """
        inputs = {
            'open': input_data[:,0],
            'high': input_data[:,1],
            'low': input_data[:,2],
            'close': input_data[:,3],
            'volume': input_data[:,4],
       }

        ema = EMA(inputs)
        _, _, macd = MACD(inputs)
        bbands = BBANDS(inputs)
        bbands = bbands[0]-bbands[1]
        rsi = RSI(inputs)/100
        mom = MOM(inputs)
        atr = ATR(inputs)
        adx = ADX(inputs)/100
        ultosc = ULTOSC(inputs)/100

        final_column = np.column_stack(([input_data, ema, macd, bbands, rsi, mom, atr, adx, ultosc]))
        return final_column[~np.isnan(final_column).any(axis=1)]

    def next_batch(self, batch_size, n_predictions, train=True):
        if train:
            # Choose random company index
            random_company = np.random.randint(len(self.training_data['Name'].unique()))
            # Get random_company data
            company_data = self.grouped_training_stocks[random_company][1]

        else:
            # Choose random company index
            random_company = np.random.randint(len(self.test_data['Name'].unique()))
            # Get random_company data
            company_data = self.grouped_test_stocks[random_company][1]

        # Select a random index to pick a batch
        random_index = np.random.randint(0, company_data.shape[0]-batch_size)

        # Prepare the batch data and turn it into an array.[['open', 'high', 'low', 'close', 'volume']*batch_size]
        batch_data_X = np.array(company_data.iloc[random_index:random_index+batch_size-n_predictions,1:6])
        batch_data_X = self.get_indicators(batch_data_X)

        # Prepare Y data ('close' values)
        batch_data_Y = np.array(company_data.iloc[random_index+batch_size-n_predictions:random_index+batch_size,4])

        return batch_data_X, batch_data_Y


# Prepare data and test batch
bh = BatchHelper(stock_prices)
train_data, test_data = bh.prepare_data()


# ---------------------------- #
#     DENSE NEURAL NETWORK     #
# ---------------------------- #
# PARAMETERS
n_predictions = 1
batch_size = 70
n_columns = 13
epochs = 500
n_steps = 50
n_neurons1 = 100
n_neurons2 = 70
n_neurons3 = 40
n_neurons4 = 15

# Avoid errors by resetting the Graph
tf.reset_default_graph()

# PLACEHOLDERS
X = tf.placeholder(tf.float32, [batch_size-n_predictions-33, n_columns])
Y = tf.placeholder(tf.float32, [n_predictions]) # Actual/correct close prices
step = tf.placeholder(tf.float32)

# BIAS AND WEIGHT INITIALIZERS
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()

# MODEL ARCHITECTURE
W1 = tf.Variable(weight_initializer([n_columns, n_neurons1]))
B1 = tf.Variable(bias_initializer([n_neurons1]))
W2 = tf.Variable(weight_initializer([n_neurons1, n_neurons2]))
B2 = tf.Variable(bias_initializer([n_neurons2]))
W3 = tf.Variable(weight_initializer([n_neurons2, n_neurons3]))
B3 = tf.Variable(bias_initializer([n_neurons3]))
W4 = tf.Variable(weight_initializer([n_neurons3, n_neurons4]))
B4 = tf.Variable(bias_initializer([n_neurons4]))
W_out = tf.Variable(weight_initializer([n_neurons4, n_predictions]))
B_out = tf.Variable(bias_initializer([n_predictions]))

# LAYERS
hidden1 = tf.nn.sigmoid(tf.matmul(X, W1) + B1)
hidden2 = tf.nn.sigmoid(tf.matmul(hidden1, W2) + B2)
hidden3 = tf.nn.sigmoid(tf.matmul(hidden2, W3) + B3)
hidden4 = tf.nn.sigmoid(tf.matmul(hidden3, W4) + B4)
Y_pred = tf.nn.sigmoid(tf.matmul(hidden4, W_out) + B_out)

# COST AND OPTIMIZER
cost = tf.reduce_mean(tf.squared_difference(Y_pred, Y))
learning_rate = 0.007 * (0.996**step)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

# SESSION
init = tf.global_variables_initializer()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    writer = tf.summary.FileWriter("E:\Code\Anaconda\Learning\Stock\output", sess.graph)
    sess.run(init)

    for e in range(epochs):
        print(f"EPOCH {e}", end=' ')
        avg_mse = []
        for s in range(n_steps):
            for i in range(5):
                try:
                    batch_X, batch_Y = bh.next_batch(batch_size, n_predictions)
                except ValueError:
                    continue
                break
            sess.run(optimizer, feed_dict={X:batch_X, Y: batch_Y, step: e})
            avg_mse.append(cost.eval(feed_dict={X:batch_X, Y: batch_Y, step: e}))

        # Print average MSE
        avg_mse = np.mean(avg_mse)
        print(f"\tAvg MSE: {avg_mse}".format(), end=' ')

        # Test the model
        print(f"\tTesting:", end=' ')
        batch_X, batch_Y = bh.next_batch(batch_size, n_predictions, train=False)
        print(sess.run(cost, feed_dict={X:batch_X, Y: batch_Y}))

    writer.close()
