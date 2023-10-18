import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest
from sklearn.preprocessing import LabelEncoder

pd.set_option("display.width", 500)
pd.set_option("display.max_columns", None)

df = pd.read_csv("website_ab_test.csv")

# Theme: dark or light
# Click-Through Rate: The proportion of the users who click on links or buttons on the website
# Conversion Rate: The percentage of users who signed up on the platform after visiting for the first time
# Bounce Rate: The percentage of users who leave the website without further interaction after visiting a single page
# Scroll Depth: The depth to which users scroll through the website pages
# Age: The age of the user
# Location: The location of the user
# Session Duration: The duration of the user’s session on the website
# Purchases: Whether the user purchased the book (Yes/No)
# Added_to_Cart: Whether the user added books to the cart (Yes/No)

df.head()
df.info()
df.isnull().sum()
df.describe().T

# Purchases 1-0 şeklinde dönüştürelim
df["Purchases"].value_counts()
# Yes    517
# No     483

label_encoder = LabelEncoder()
df["Purchases"] = label_encoder.fit_transform(df["Purchases"])
# 1    517
# 0    483

# Conversion Rate ve Purchases değerlerinin incelemesi yapılacak
df.groupby("Theme").agg({"Conversion Rate": "mean",
                         "Purchases":"mean"})
#              Conversion Rate  Purchases
# Theme
# Dark Theme          0.251282   0.503891
# Light Theme         0.255459   0.530864

# Conversion Rate Dark ve Light Temalar arasında oldukça az bir fark var. Bu farkın istatiksel olarak
# anlamlı olup olmadığını inceleyeceğiz.
#Bunun için Hipotez testlerinden yararlanacağız ve p-value değerini inceleyeceğiz.

# 1. Hipotezler Kurulacak
# 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı (shapiro)
#   - 2. Varyans Homojenliği (levene)
# 3. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi
# 4. p-value değerine göre sonuçları yorumla

# Hipotezler
# H0 = Dark Theme ve Light Theme'lerin Conversion Rate'lerinde istatistiksel olarak anlamlı bir fark YOKTUR.
# H1 =  Dark Theme ve Light Theme'lerin Conversion Rate'lerinde istatistiksel olarak anlamlı bir fark VARDIR.

# Datasetimizin normallik dağıllımı incelenmeli (Light ve Dark Temalar için ayrı ayrı)
df_light_cr =df[df["Theme"]=="Light Theme"]
df_light_cr = df_light_cr.reset_index()

test_stat, pvalue = shapiro(df_light_cr["Conversion Rate"])
print("Test stat = %.4f, p-value = %.4f" % (test_stat, pvalue))
# Test stat = 0.9564, p-value = 0.0000 ----> p-value < 0.05 ----> H0 Red Edilir. Normallik varsayımı sağlanmamıştır.

df_dark_cr = df[df["Theme"]=="Dark Theme"]
df_dark_cr = df_dark_cr.reset_index()

test_stat, pvalue = shapiro(df_dark_cr["Conversion Rate"])
print("Test stat = %.4f, p-value = %.4f" % (test_stat, pvalue))
# Test stat = 0.9564, p-value = 0.0000 ----> p-value < 0.05 ----> H0 Red Edilir. Normallik varsayımı sağlanmamıştır.

# Normal Dağılım Varsayımı sağlanmadığı için Non-parametrik(mannwhitneyu) testimize geçiyoruz.

test_stat, pvalue = mannwhitneyu(df.loc[df["Theme"]=="Light Theme", "Conversion Rate"],
                                 df.loc[df["Theme"]=="Dark Theme", "Conversion Rate"])
print("Test stat = %.4f, p-value = %.4f" % (test_stat, pvalue))
# Test stat = 127207.0000, p-value = 0.6137 ----> p-value > 0.05 ----> H0 Red Edilemez.

#### H0 hipotezimiz geçerlidir ####
#### Dark Theme ve Light Theme'lerin Conversion Rate'lerinde istatistiksel olarak anlamlı bir fark YOKTUR ####



# Purchases işlemleri için hipotez testlerimizi uygulayalım.

#              Conversion Rate  Purchases
# Theme
# Dark Theme          0.251282   0.503891
# Light Theme         0.255459   0.530864

# 1. Hipotezler Kurulacak
# 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı (shapiro)
#   - 2. Varyans Homojenliği (levene)
# 3. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi
# 4. p-value değerine göre sonuçları yorumla


# Hipotezler
# H0 = Dark Theme ve Light Theme'lerin Satın alma değerlerinde istatistiksel olarak anlamlı bir fark YOKTUR.
# H1 =  Dark Theme ve Light Theme'lerin Satın alma değerlerinde istatistiksel olarak anlamlı bir fark VARDIR.

# Datasetimizin normallik dağıllımı incelenmeli (Light ve Dark Temalar için ayrı ayrı)
df_light_p = df[df["Theme"] == "Light Theme"]
df_light_p = df_light_p.reset_index()

test_stat, pvalue = shapiro(df_light_p["Purchases"])
print("Test stat = %.4f, p-value = %.4f" % (test_stat, pvalue))
# Test stat = 0.6352, p-value = 0.0000 ----> p-value < 0.05 ----> H0 Red Edilir. Normallik varsayımı sağlanmamıştır.

df_dark_p = df[df["Theme"] == "Dark Theme"]
df_dark_p = df_dark_cr.reset_index()

test_stat, pvalue = shapiro(df_dark_cr["Purchases"])
print("Test stat = %.4f, p-value = %.4f" % (test_stat, pvalue))
# Test stat = 0.6365, p-value = 0.0000 ----> p-value < 0.05 ----> H0 Red Edilir. Normallik varsayımı sağlanmamıştır.

# Normal Dağılım Varsayımı sağlanmadığı için Non-parametrik(mannwhitneyu) testimize geçiyoruz.

test_stat, pvalue = mannwhitneyu(df.loc[df["Theme"] == "Light Theme", "Purchases"],
                                 df.loc[df["Theme"] == "Dark Theme", "Purchases"])
print("Test stat = %.4f, p-value = %.4f" % (test_stat, pvalue))
# Test stat = 128271.0000, p-value = 0.3939 ----> p-value > 0.05 ----> H0 Red Edilemez.

#### H0 hipotezimiz geçerlidir ####
#### Dark Theme ve Light Theme'lerin satışlarında istatistiksel olarak anlamlı bir fark YOKTUR ####









