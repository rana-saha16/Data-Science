{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "49f948a0",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "d50a8331",
   "metadata": {},
   "outputs": [],
   "source": [
    "data=pd.read_csv('Test_Sales_LinReg.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "85011075",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Republic</th>\n",
       "      <th>NDTV</th>\n",
       "      <th>TV5</th>\n",
       "      <th>TV9</th>\n",
       "      <th>AajTak</th>\n",
       "      <th>sales</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>8.7</td>\n",
       "      <td>48.9</td>\n",
       "      <td>4.0</td>\n",
       "      <td>75.0</td>\n",
       "      <td>49.0</td>\n",
       "      <td>7.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>57.5</td>\n",
       "      <td>32.8</td>\n",
       "      <td>65.9</td>\n",
       "      <td>23.5</td>\n",
       "      <td>57.5</td>\n",
       "      <td>11.8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>120.2</td>\n",
       "      <td>19.6</td>\n",
       "      <td>7.2</td>\n",
       "      <td>11.6</td>\n",
       "      <td>18.5</td>\n",
       "      <td>13.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>8.6</td>\n",
       "      <td>2.1</td>\n",
       "      <td>46.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>2.6</td>\n",
       "      <td>4.8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>199.8</td>\n",
       "      <td>2.6</td>\n",
       "      <td>52.9</td>\n",
       "      <td>21.2</td>\n",
       "      <td>2.9</td>\n",
       "      <td>10.6</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Republic  NDTV   TV5   TV9  AajTak  sales\n",
       "0       8.7  48.9   4.0  75.0    49.0    7.2\n",
       "1      57.5  32.8  65.9  23.5    57.5   11.8\n",
       "2     120.2  19.6   7.2  11.6    18.5   13.2\n",
       "3       8.6   2.1  46.0   1.0     2.6    4.8\n",
       "4     199.8   2.6  52.9  21.2     2.9   10.6"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "0213b25e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(305, 6)"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "fd188068",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Republic</th>\n",
       "      <th>NDTV</th>\n",
       "      <th>TV5</th>\n",
       "      <th>TV9</th>\n",
       "      <th>AajTak</th>\n",
       "      <th>sales</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>295.000000</td>\n",
       "      <td>300.000000</td>\n",
       "      <td>305.000000</td>\n",
       "      <td>297.000000</td>\n",
       "      <td>300.000000</td>\n",
       "      <td>305.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>148.136610</td>\n",
       "      <td>22.341333</td>\n",
       "      <td>29.459344</td>\n",
       "      <td>28.862626</td>\n",
       "      <td>23.517967</td>\n",
       "      <td>13.811475</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>87.330161</td>\n",
       "      <td>14.781927</td>\n",
       "      <td>20.290023</td>\n",
       "      <td>21.411180</td>\n",
       "      <td>15.853789</td>\n",
       "      <td>5.192185</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.700000</td>\n",
       "      <td>0.300000</td>\n",
       "      <td>0.300000</td>\n",
       "      <td>0.300000</td>\n",
       "      <td>0.300000</td>\n",
       "      <td>1.600000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>74.050000</td>\n",
       "      <td>9.125000</td>\n",
       "      <td>15.900000</td>\n",
       "      <td>10.900000</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>10.300000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>149.800000</td>\n",
       "      <td>21.050000</td>\n",
       "      <td>26.200000</td>\n",
       "      <td>23.500000</td>\n",
       "      <td>21.300000</td>\n",
       "      <td>12.800000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>221.450000</td>\n",
       "      <td>35.650000</td>\n",
       "      <td>39.600000</td>\n",
       "      <td>43.000000</td>\n",
       "      <td>36.900000</td>\n",
       "      <td>17.200000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>296.400000</td>\n",
       "      <td>49.400000</td>\n",
       "      <td>114.000000</td>\n",
       "      <td>114.000000</td>\n",
       "      <td>75.500000</td>\n",
       "      <td>27.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         Republic        NDTV         TV5         TV9      AajTak       sales\n",
       "count  295.000000  300.000000  305.000000  297.000000  300.000000  305.000000\n",
       "mean   148.136610   22.341333   29.459344   28.862626   23.517967   13.811475\n",
       "std     87.330161   14.781927   20.290023   21.411180   15.853789    5.192185\n",
       "min      0.700000    0.300000    0.300000    0.300000    0.300000    1.600000\n",
       "25%     74.050000    9.125000   15.900000   10.900000   10.000000   10.300000\n",
       "50%    149.800000   21.050000   26.200000   23.500000   21.300000   12.800000\n",
       "75%    221.450000   35.650000   39.600000   43.000000   36.900000   17.200000\n",
       "max    296.400000   49.400000  114.000000  114.000000   75.500000   27.000000"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "e0d530ca",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 305 entries, 0 to 304\n",
      "Data columns (total 6 columns):\n",
      " #   Column    Non-Null Count  Dtype  \n",
      "---  ------    --------------  -----  \n",
      " 0   Republic  295 non-null    float64\n",
      " 1   NDTV      300 non-null    float64\n",
      " 2   TV5       305 non-null    float64\n",
      " 3   TV9       297 non-null    float64\n",
      " 4   AajTak    300 non-null    float64\n",
      " 5   sales     305 non-null    float64\n",
      "dtypes: float64(6)\n",
      "memory usage: 14.4 KB\n"
     ]
    }
   ],
   "source": [
    "data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "fb63fe14",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Republic    10\n",
       "NDTV         5\n",
       "TV5          0\n",
       "TV9          8\n",
       "AajTak       5\n",
       "sales        0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "c35b2069",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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m8tbRERkH9u06OC+Jjcx3oiI7YB7Ijcorz/UJXx+d0unOxtCRKQA27To4L4mNzHeiIjtgHsiNyi/cOIY4fOzJxe6syFERAqwTYsO7mtyE+ONiNgOuCdyg/LLZpSiMC9X97nCvFxWEySiQGGbFh3c1+QmxhsRsR1wT+QG5QBQvXJ+WoBpVQSJiIKGbVp0cF+TmxhvRMR2wB2RXBJtUlEe6tYswdbGNuxu6eR6e0QUaGzTooP7mtzEeCMitgPuiOSgXLOgvIRBRUShwTYtOrivyU2MNyJiO+CsSE5fJyIiIiIiIvIDDsqJiIiIiIiIPMJBOREREREREZFHOCgnIiIiIiIi8ggH5UREREREREQe4aCciIiIiIiIyCMclBMRERERERF5hINyIiIiIiIiIo9wUE5ERERERETkEQ7KiYiIiIiI6P9v796DojrPMIA/y3VRdFsLZkWMrqKigpjiLaSCrRTUUKtoio7WCzttQKEhSWNaTQrWRLBTnapJ0BhuadqsNKJGp6OSIAhaLSVsisELEfESl6FxMCIVFPbrHwzbbEDYJSsHznl+Mzsj5/q9e97Xs++es7skETblRERERERERBJhU05EREREREQkETblRERERERERBJhU05EREREREQkERepB/CoCSEAAHfu3JF4JKRE7XnXnoe9jflPUmMNkJIx/0npWAOkZPbkv+yb8oaGBgDAiBEjJB4JKVlDQwM0Go0k+wWY/yQ91gApGfOflI41QEpmS/6rhFRvXfUSs9mMmzdvYtCgQVCpVFbz7ty5gxEjRuD69esYPHiwRCMkQL7HQgiBhoYG+Pj4wMmp9z8twvxvo6RYgb4VL2ug72L8jz5+5n/fxfh7J/6+WgNKP/59gRKOgT35L/sr5U5OTvD19e1ymcGDB8s2GfobOR4LKd4Zbsf8t6akWIG+Ey9roG9j/I82fuZ/38b4H338fbkGlH78+wK5HwNb859f9EZEREREREQkETblRERERERERBJRdFPu7u6O5ORkuLu7Sz0UxeOx6H1Kes6VFCugvHh7SunPE+Nn/Iyf8TN+ZcbfF/AYWJP9F70RERERERER9VWKvlJOREREREREJCU25UREREREREQSYVNOREREREREJBE25UREREREREQSUWxT/tZbb0Gn00GtViM4OBjFxcVSD0lWUlNTMW3aNAwaNAhDhw7FwoUL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",
      "text/plain": [
       "<Figure size 1200x600 with 5 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig,axs=plt.subplots(1,5)\n",
    "data.plot(kind='scatter', x='Republic',y='sales', ax=axs[0],figsize=(12, 6))\n",
    "data.plot(kind='scatter',x ='NDTV', y='sales', ax=axs[1])\n",
    "data.plot(kind='scatter', x='TV5', y='sales',ax=axs[2])\n",
    "data.plot(kind='scatter', x='TV9', y='sales',ax=axs[3])\n",
    "data.plot(kind='scatter', x='AajTak', y='sales',ax=axs[4])\n",
    "fig.savefig('testdata.jpg')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "e95c902e",
   "metadata": {},
   "outputs": [],
   "source": [
    "x=data[['Republic','NDTV','TV5','TV9','AajTak']]\n",
    "y=data.sales"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "cbaf9c7b",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LinearRegression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "bca13e86",
   "metadata": {},
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "Input X contains NaN.\nLinearRegression does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[45], line 2\u001b[0m\n\u001b[0;32m      1\u001b[0m lm\u001b[38;5;241m=\u001b[39mLinearRegression()\n\u001b[1;32m----> 2\u001b[0m lm\u001b[38;5;241m.\u001b[39mfit(x,y)\n",
      "File \u001b[1;32mC:\\ProgramData\\anaconda3\\Lib\\site-packages\\sklearn\\base.py:1151\u001b[0m, in \u001b[0;36m_fit_context.<locals>.decorator.<locals>.wrapper\u001b[1;34m(estimator, *args, **kwargs)\u001b[0m\n\u001b[0;32m   1144\u001b[0m     estimator\u001b[38;5;241m.\u001b[39m_validate_params()\n\u001b[0;32m   1146\u001b[0m \u001b[38;5;28;01mwith\u001b[39;00m config_context(\n\u001b[0;32m   1147\u001b[0m     skip_parameter_validation\u001b[38;5;241m=\u001b[39m(\n\u001b[0;32m   1148\u001b[0m         prefer_skip_nested_validation \u001b[38;5;129;01mor\u001b[39;00m global_skip_validation\n\u001b[0;32m   1149\u001b[0m     )\n\u001b[0;32m   1150\u001b[0m ):\n\u001b[1;32m-> 1151\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m fit_method(estimator, \u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n",
      "File \u001b[1;32mC:\\ProgramData\\anaconda3\\Lib\\site-packages\\sklearn\\linear_model\\_base.py:678\u001b[0m, in \u001b[0;36mLinearRegression.fit\u001b[1;34m(self, X, y, sample_weight)\u001b[0m\n\u001b[0;32m    674\u001b[0m n_jobs_ \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mn_jobs\n\u001b[0;32m    676\u001b[0m accept_sparse \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mFalse\u001b[39;00m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mpositive \u001b[38;5;28;01melse\u001b[39;00m [\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mcsr\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mcsc\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mcoo\u001b[39m\u001b[38;5;124m\"\u001b[39m]\n\u001b[1;32m--> 678\u001b[0m X, y \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_validate_data(\n\u001b[0;32m    679\u001b[0m     X, y, accept_sparse\u001b[38;5;241m=\u001b[39maccept_sparse, y_numeric\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mTrue\u001b[39;00m, multi_output\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mTrue\u001b[39;00m\n\u001b[0;32m    680\u001b[0m )\n\u001b[0;32m    682\u001b[0m has_sw \u001b[38;5;241m=\u001b[39m sample_weight \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[0;32m    683\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m has_sw:\n",
      "File \u001b[1;32mC:\\ProgramData\\anaconda3\\Lib\\site-packages\\sklearn\\base.py:621\u001b[0m, in \u001b[0;36mBaseEstimator._validate_data\u001b[1;34m(self, X, y, reset, validate_separately, cast_to_ndarray, **check_params)\u001b[0m\n\u001b[0;32m    619\u001b[0m         y \u001b[38;5;241m=\u001b[39m check_array(y, input_name\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124my\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mcheck_y_params)\n\u001b[0;32m    620\u001b[0m     \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m--> 621\u001b[0m         X, y \u001b[38;5;241m=\u001b[39m check_X_y(X, y, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mcheck_params)\n\u001b[0;32m    622\u001b[0m     out \u001b[38;5;241m=\u001b[39m X, y\n\u001b[0;32m    624\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m no_val_X \u001b[38;5;129;01mand\u001b[39;00m check_params\u001b[38;5;241m.\u001b[39mget(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mensure_2d\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;28;01mTrue\u001b[39;00m):\n",
      "File \u001b[1;32mC:\\ProgramData\\anaconda3\\Lib\\site-packages\\sklearn\\utils\\validation.py:1147\u001b[0m, in \u001b[0;36mcheck_X_y\u001b[1;34m(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)\u001b[0m\n\u001b[0;32m   1142\u001b[0m         estimator_name \u001b[38;5;241m=\u001b[39m _check_estimator_name(estimator)\n\u001b[0;32m   1143\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\n\u001b[0;32m   1144\u001b[0m         \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;132;01m{\u001b[39;00mestimator_name\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m requires y to be passed, but the target y is None\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m   1145\u001b[0m     )\n\u001b[1;32m-> 1147\u001b[0m X \u001b[38;5;241m=\u001b[39m check_array(\n\u001b[0;32m   1148\u001b[0m     X,\n\u001b[0;32m   1149\u001b[0m     accept_sparse\u001b[38;5;241m=\u001b[39maccept_sparse,\n\u001b[0;32m   1150\u001b[0m     accept_large_sparse\u001b[38;5;241m=\u001b[39maccept_large_sparse,\n\u001b[0;32m   1151\u001b[0m     dtype\u001b[38;5;241m=\u001b[39mdtype,\n\u001b[0;32m   1152\u001b[0m     order\u001b[38;5;241m=\u001b[39morder,\n\u001b[0;32m   1153\u001b[0m     copy\u001b[38;5;241m=\u001b[39mcopy,\n\u001b[0;32m   1154\u001b[0m     force_all_finite\u001b[38;5;241m=\u001b[39mforce_all_finite,\n\u001b[0;32m   1155\u001b[0m     ensure_2d\u001b[38;5;241m=\u001b[39mensure_2d,\n\u001b[0;32m   1156\u001b[0m     allow_nd\u001b[38;5;241m=\u001b[39mallow_nd,\n\u001b[0;32m   1157\u001b[0m     ensure_min_samples\u001b[38;5;241m=\u001b[39mensure_min_samples,\n\u001b[0;32m   1158\u001b[0m     ensure_min_features\u001b[38;5;241m=\u001b[39mensure_min_features,\n\u001b[0;32m   1159\u001b[0m     estimator\u001b[38;5;241m=\u001b[39mestimator,\n\u001b[0;32m   1160\u001b[0m     input_name\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mX\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[0;32m   1161\u001b[0m )\n\u001b[0;32m   1163\u001b[0m y \u001b[38;5;241m=\u001b[39m _check_y(y, multi_output\u001b[38;5;241m=\u001b[39mmulti_output, y_numeric\u001b[38;5;241m=\u001b[39my_numeric, estimator\u001b[38;5;241m=\u001b[39mestimator)\n\u001b[0;32m   1165\u001b[0m check_consistent_length(X, y)\n",
      "File \u001b[1;32mC:\\ProgramData\\anaconda3\\Lib\\site-packages\\sklearn\\utils\\validation.py:959\u001b[0m, in \u001b[0;36mcheck_array\u001b[1;34m(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)\u001b[0m\n\u001b[0;32m    953\u001b[0m         \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\n\u001b[0;32m    954\u001b[0m             \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mFound array with dim \u001b[39m\u001b[38;5;132;01m%d\u001b[39;00m\u001b[38;5;124m. \u001b[39m\u001b[38;5;132;01m%s\u001b[39;00m\u001b[38;5;124m expected <= 2.\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m    955\u001b[0m             \u001b[38;5;241m%\u001b[39m (array\u001b[38;5;241m.\u001b[39mndim, estimator_name)\n\u001b[0;32m    956\u001b[0m         )\n\u001b[0;32m    958\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m force_all_finite:\n\u001b[1;32m--> 959\u001b[0m         _assert_all_finite(\n\u001b[0;32m    960\u001b[0m             array,\n\u001b[0;32m    961\u001b[0m             input_name\u001b[38;5;241m=\u001b[39minput_name,\n\u001b[0;32m    962\u001b[0m             estimator_name\u001b[38;5;241m=\u001b[39mestimator_name,\n\u001b[0;32m    963\u001b[0m             allow_nan\u001b[38;5;241m=\u001b[39mforce_all_finite \u001b[38;5;241m==\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mallow-nan\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[0;32m    964\u001b[0m         )\n\u001b[0;32m    966\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m ensure_min_samples \u001b[38;5;241m>\u001b[39m \u001b[38;5;241m0\u001b[39m:\n\u001b[0;32m    967\u001b[0m     n_samples \u001b[38;5;241m=\u001b[39m _num_samples(array)\n",
      "File \u001b[1;32mC:\\ProgramData\\anaconda3\\Lib\\site-packages\\sklearn\\utils\\validation.py:124\u001b[0m, in \u001b[0;36m_assert_all_finite\u001b[1;34m(X, allow_nan, msg_dtype, estimator_name, input_name)\u001b[0m\n\u001b[0;32m    121\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m first_pass_isfinite:\n\u001b[0;32m    122\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m\n\u001b[1;32m--> 124\u001b[0m _assert_all_finite_element_wise(\n\u001b[0;32m    125\u001b[0m     X,\n\u001b[0;32m    126\u001b[0m     xp\u001b[38;5;241m=\u001b[39mxp,\n\u001b[0;32m    127\u001b[0m     allow_nan\u001b[38;5;241m=\u001b[39mallow_nan,\n\u001b[0;32m    128\u001b[0m     msg_dtype\u001b[38;5;241m=\u001b[39mmsg_dtype,\n\u001b[0;32m    129\u001b[0m     estimator_name\u001b[38;5;241m=\u001b[39mestimator_name,\n\u001b[0;32m    130\u001b[0m     input_name\u001b[38;5;241m=\u001b[39minput_name,\n\u001b[0;32m    131\u001b[0m )\n",
      "File \u001b[1;32mC:\\ProgramData\\anaconda3\\Lib\\site-packages\\sklearn\\utils\\validation.py:173\u001b[0m, in \u001b[0;36m_assert_all_finite_element_wise\u001b[1;34m(X, xp, allow_nan, msg_dtype, estimator_name, input_name)\u001b[0m\n\u001b[0;32m    156\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m estimator_name \u001b[38;5;129;01mand\u001b[39;00m input_name \u001b[38;5;241m==\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mX\u001b[39m\u001b[38;5;124m\"\u001b[39m \u001b[38;5;129;01mand\u001b[39;00m has_nan_error:\n\u001b[0;32m    157\u001b[0m     \u001b[38;5;66;03m# Improve the error message on how to handle missing values in\u001b[39;00m\n\u001b[0;32m    158\u001b[0m     \u001b[38;5;66;03m# scikit-learn.\u001b[39;00m\n\u001b[0;32m    159\u001b[0m     msg_err \u001b[38;5;241m+\u001b[39m\u001b[38;5;241m=\u001b[39m (\n\u001b[0;32m    160\u001b[0m         \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;132;01m{\u001b[39;00mestimator_name\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m does not accept missing values\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m    161\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m encoded as NaN natively. For supervised learning, you might want\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m    171\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m#estimators-that-handle-nan-values\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m    172\u001b[0m     )\n\u001b[1;32m--> 173\u001b[0m \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(msg_err)\n",
      "\u001b[1;31mValueError\u001b[0m: Input X contains NaN.\nLinearRegression does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values"
     ]
    }
   ],
   "source": [
    "lm=LinearRegression()\n",
    "lm.fit(x,y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "4eb876c7",
   "metadata": {},
   "outputs": [
    {
     "ename": "AttributeError",
     "evalue": "'LinearRegression' object has no attribute 'intercept_'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[44], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m \u001b[38;5;28mprint\u001b[39m(lm\u001b[38;5;241m.\u001b[39mintercept_)\n\u001b[0;32m      2\u001b[0m \u001b[38;5;28mprint\u001b[39m(lm\u001b[38;5;241m.\u001b[39mcoef_)\n",
      "\u001b[1;31mAttributeError\u001b[0m: 'LinearRegression' object has no attribute 'intercept_'"
     ]
    }
   ],
   "source": [
    "print(lm.intercept_)\n",
    "print(lm.coef_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "a549cbbe",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Republic</th>\n",
       "      <th>NDTV</th>\n",
       "      <th>TV5</th>\n",
       "      <th>TV9</th>\n",
       "      <th>AajTak</th>\n",
       "      <th>sales</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>8.7</td>\n",
       "      <td>48.9</td>\n",
       "      <td>4.0</td>\n",
       "      <td>75.0</td>\n",
       "      <td>49.0</td>\n",
       "      <td>7.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>57.5</td>\n",
       "      <td>32.8</td>\n",
       "      <td>65.9</td>\n",
       "      <td>23.5</td>\n",
       "      <td>57.5</td>\n",
       "      <td>11.8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>120.2</td>\n",
       "      <td>19.6</td>\n",
       "      <td>7.2</td>\n",
       "      <td>11.6</td>\n",
       "      <td>18.5</td>\n",
       "      <td>13.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>8.6</td>\n",
       "      <td>2.1</td>\n",
       "      <td>46.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>2.6</td>\n",
       "      <td>4.8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>199.8</td>\n",
       "      <td>2.6</td>\n",
       "      <td>52.9</td>\n",
       "      <td>21.2</td>\n",
       "      <td>2.9</td>\n",
       "      <td>10.6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>300</th>\n",
       "      <td>286.0</td>\n",
       "      <td>13.9</td>\n",
       "      <td>35.2</td>\n",
       "      <td>3.7</td>\n",
       "      <td>13.9</td>\n",
       "      <td>15.9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>301</th>\n",
       "      <td>18.7</td>\n",
       "      <td>12.1</td>\n",
       "      <td>23.7</td>\n",
       "      <td>23.4</td>\n",
       "      <td>18.7</td>\n",
       "      <td>6.7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>302</th>\n",
       "      <td>39.5</td>\n",
       "      <td>41.1</td>\n",
       "      <td>17.6</td>\n",
       "      <td>5.8</td>\n",
       "      <td>39.5</td>\n",
       "      <td>10.8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>303</th>\n",
       "      <td>75.5</td>\n",
       "      <td>10.8</td>\n",
       "      <td>8.3</td>\n",
       "      <td>6.0</td>\n",
       "      <td>75.5</td>\n",
       "      <td>9.9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>304</th>\n",
       "      <td>17.2</td>\n",
       "      <td>4.1</td>\n",
       "      <td>30.0</td>\n",
       "      <td>31.6</td>\n",
       "      <td>17.2</td>\n",
       "      <td>5.9</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>305 rows × 6 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Republic  NDTV   TV5   TV9  AajTak  sales\n",
       "0         8.7  48.9   4.0  75.0    49.0    7.2\n",
       "1        57.5  32.8  65.9  23.5    57.5   11.8\n",
       "2       120.2  19.6   7.2  11.6    18.5   13.2\n",
       "3         8.6   2.1  46.0   1.0     2.6    4.8\n",
       "4       199.8   2.6  52.9  21.2     2.9   10.6\n",
       "..        ...   ...   ...   ...     ...    ...\n",
       "300     286.0  13.9  35.2   3.7    13.9   15.9\n",
       "301      18.7  12.1  23.7  23.4    18.7    6.7\n",
       "302      39.5  41.1  17.6   5.8    39.5   10.8\n",
       "303      75.5  10.8   8.3   6.0    75.5    9.9\n",
       "304      17.2   4.1  30.0  31.6    17.2    5.9\n",
       "\n",
       "[305 rows x 6 columns]"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df=pd.DataFrame(data)\n",
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "f6c11510",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import r2_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "006c6d49",
   "metadata": {},
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "The feature names should match those that were passed during fit.\nFeature names unseen at fit time:\n- AajTak\n- NDTV\n- TV5\n- TV9\n",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[42], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m predicted_sales\u001b[38;5;241m=\u001b[39mlm\u001b[38;5;241m.\u001b[39mpredict(x)\n",
      "File \u001b[1;32mC:\\ProgramData\\anaconda3\\Lib\\site-packages\\sklearn\\linear_model\\_base.py:386\u001b[0m, in \u001b[0;36mLinearModel.predict\u001b[1;34m(self, X)\u001b[0m\n\u001b[0;32m    372\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mpredict\u001b[39m(\u001b[38;5;28mself\u001b[39m, X):\n\u001b[0;32m    373\u001b[0m \u001b[38;5;250m    \u001b[39m\u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[0;32m    374\u001b[0m \u001b[38;5;124;03m    Predict using the linear model.\u001b[39;00m\n\u001b[0;32m    375\u001b[0m \n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m    384\u001b[0m \u001b[38;5;124;03m        Returns predicted values.\u001b[39;00m\n\u001b[0;32m    385\u001b[0m \u001b[38;5;124;03m    \"\"\"\u001b[39;00m\n\u001b[1;32m--> 386\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_decision_function(X)\n",
      "File \u001b[1;32mC:\\ProgramData\\anaconda3\\Lib\\site-packages\\sklearn\\linear_model\\_base.py:369\u001b[0m, in \u001b[0;36mLinearModel._decision_function\u001b[1;34m(self, X)\u001b[0m\n\u001b[0;32m    366\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21m_decision_function\u001b[39m(\u001b[38;5;28mself\u001b[39m, X):\n\u001b[0;32m    367\u001b[0m     check_is_fitted(\u001b[38;5;28mself\u001b[39m)\n\u001b[1;32m--> 369\u001b[0m     X \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_validate_data(X, accept_sparse\u001b[38;5;241m=\u001b[39m[\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mcsr\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mcsc\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mcoo\u001b[39m\u001b[38;5;124m\"\u001b[39m], reset\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mFalse\u001b[39;00m)\n\u001b[0;32m    370\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m safe_sparse_dot(X, \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mcoef_\u001b[38;5;241m.\u001b[39mT, dense_output\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mTrue\u001b[39;00m) \u001b[38;5;241m+\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mintercept_\n",
      "File \u001b[1;32mC:\\ProgramData\\anaconda3\\Lib\\site-packages\\sklearn\\base.py:579\u001b[0m, in \u001b[0;36mBaseEstimator._validate_data\u001b[1;34m(self, X, y, reset, validate_separately, cast_to_ndarray, **check_params)\u001b[0m\n\u001b[0;32m    508\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21m_validate_data\u001b[39m(\n\u001b[0;32m    509\u001b[0m     \u001b[38;5;28mself\u001b[39m,\n\u001b[0;32m    510\u001b[0m     X\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mno_validation\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m    515\u001b[0m     \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mcheck_params,\n\u001b[0;32m    516\u001b[0m ):\n\u001b[0;32m    517\u001b[0m \u001b[38;5;250m    \u001b[39m\u001b[38;5;124;03m\"\"\"Validate input data and set or check the `n_features_in_` attribute.\u001b[39;00m\n\u001b[0;32m    518\u001b[0m \n\u001b[0;32m    519\u001b[0m \u001b[38;5;124;03m    Parameters\u001b[39;00m\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m    577\u001b[0m \u001b[38;5;124;03m        validated.\u001b[39;00m\n\u001b[0;32m    578\u001b[0m \u001b[38;5;124;03m    \"\"\"\u001b[39;00m\n\u001b[1;32m--> 579\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_check_feature_names(X, reset\u001b[38;5;241m=\u001b[39mreset)\n\u001b[0;32m    581\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m y \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m \u001b[38;5;129;01mand\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_get_tags()[\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mrequires_y\u001b[39m\u001b[38;5;124m\"\u001b[39m]:\n\u001b[0;32m    582\u001b[0m         \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\n\u001b[0;32m    583\u001b[0m             \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mThis \u001b[39m\u001b[38;5;132;01m{\u001b[39;00m\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m\u001b[38;5;18m__class__\u001b[39m\u001b[38;5;241m.\u001b[39m\u001b[38;5;18m__name__\u001b[39m\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m estimator \u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m    584\u001b[0m             \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mrequires y to be passed, but the target y is None.\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m    585\u001b[0m         )\n",
      "File \u001b[1;32mC:\\ProgramData\\anaconda3\\Lib\\site-packages\\sklearn\\base.py:506\u001b[0m, in \u001b[0;36mBaseEstimator._check_feature_names\u001b[1;34m(self, X, reset)\u001b[0m\n\u001b[0;32m    501\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m missing_names \u001b[38;5;129;01mand\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m unexpected_names:\n\u001b[0;32m    502\u001b[0m     message \u001b[38;5;241m+\u001b[39m\u001b[38;5;241m=\u001b[39m (\n\u001b[0;32m    503\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mFeature names must be in the same order as they were in fit.\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m    504\u001b[0m     )\n\u001b[1;32m--> 506\u001b[0m \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(message)\n",
      "\u001b[1;31mValueError\u001b[0m: The feature names should match those that were passed during fit.\nFeature names unseen at fit time:\n- AajTak\n- NDTV\n- TV5\n- TV9\n"
     ]
    }
   ],
   "source": [
    "predicted_sales=lm.predict(x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "1da9013f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Republic</th>\n",
       "      <th>NDTV</th>\n",
       "      <th>TV5</th>\n",
       "      <th>TV9</th>\n",
       "      <th>AajTak</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>8.7</td>\n",
       "      <td>48.9</td>\n",
       "      <td>4.0</td>\n",
       "      <td>75.0</td>\n",
       "      <td>49.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>57.5</td>\n",
       "      <td>32.8</td>\n",
       "      <td>65.9</td>\n",
       "      <td>23.5</td>\n",
       "      <td>57.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>120.2</td>\n",
       "      <td>19.6</td>\n",
       "      <td>7.2</td>\n",
       "      <td>11.6</td>\n",
       "      <td>18.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>8.6</td>\n",
       "      <td>2.1</td>\n",
       "      <td>46.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>2.6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>199.8</td>\n",
       "      <td>2.6</td>\n",
       "      <td>52.9</td>\n",
       "      <td>21.2</td>\n",
       "      <td>2.9</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Republic  NDTV   TV5   TV9  AajTak\n",
       "0       8.7  48.9   4.0  75.0    49.0\n",
       "1      57.5  32.8  65.9  23.5    57.5\n",
       "2     120.2  19.6   7.2  11.6    18.5\n",
       "3       8.6   2.1  46.0   1.0     2.6\n",
       "4     199.8   2.6  52.9  21.2     2.9"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "38d16414",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'predicted_sales' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[41], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m predicted_sales\n",
      "\u001b[1;31mNameError\u001b[0m: name 'predicted_sales' is not defined"
     ]
    }
   ],
   "source": [
    "predicted_sales"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "e0de2e20",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'predicted_sales' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[46], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m r2_score(y_true\u001b[38;5;241m=\u001b[39my,y_pred\u001b[38;5;241m=\u001b[39mpredicted_sales)\n",
      "\u001b[1;31mNameError\u001b[0m: name 'predicted_sales' is not defined"
     ]
    }
   ],
   "source": [
    "r2_score(y_true=y,y_pred=predicted_sales)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "15a2efdd",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
