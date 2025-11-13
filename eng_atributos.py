import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

dataset = pd.read_csv('credit_simple.csv', sep=';')
#print(dataset.shape)
#print(dataset.head)

y = dataset['CLASSE']
x = dataset.iloc[:,:-1]

#print(x.isnull().sum())

# Solucionando o problema de valores nulos em Saldo Atual com a mediana
mediana = x['SALDO_ATUAL'].median()
x['SALDO_ATUAL'].fillna(mediana,inplace=True)
#print(x.isnull().sum())

# Solucionando o problema de valores nulos em Estado Civil com a moda
agrupado = x.groupby(['ESTADOCIVIL']).size()
x['ESTADOCIVIL'].fillna('masculino solteiro', inplace=True)
#print(x.isnull().sum())

desv = x['SALDO_ATUAL'].std()
#print(desv)

#print(x.loc[x['SALDO_ATUAL']>= 2 * desv, 'SALDO_ATUAL'])

mediana = x['SALDO_ATUAL'].median()
x.loc[x['SALDO_ATUAL']>= 2 * desv, 'SALDO_ATUAL'] = mediana
#print(x.loc[x['SALDO_ATUAL']>= 2 * desv])

x.loc[x['PROPOSITO'] == 'Eletrodomésticos','PROPOSITO'] = 'outros'
x.loc[x['PROPOSITO'] == 'qualificação','PROPOSITO'] = 'outros'

#Retira elementos menos presentes e coloca e adiciona em outra catergoria
agrupado = x.groupby(['PROPOSITO']).size()
#print(agrupado)

#Formatando Data
x['DATA'] = pd.to_datetime(x['DATA'], format='%d/%m/%Y')
#print(x['DATA'])

x['ANO'] = x['DATA'].dt.year
x['MES'] = x['DATA'].dt.month
x['DIASEMANA'] = x['DATA'].dt.day_name()
#print(x['DIASEMANA'])

#Usando o LabelEncoder
labelencolder1 = LabelEncoder()
x['ESTADOCIVIL'] = labelencolder1.fit_transform(x['ESTADOCIVIL'])
x['PROPOSITO'] = labelencolder1.fit_transform(x['PROPOSITO'])
x['DIASEMANA'] = labelencolder1.fit_transform(x['DIASEMANA'])
#print(x.head())

# Usando o OneHotEncoder
z = pd.get_dummies(x['OUTROSPLANOSPGTO'], prefix='OUTROS')
#print(z)

# Usando StandardScaler
sc = StandardScaler()
m = sc.fit_transform(x.iloc[:,0:3])
#print(m)

# Concatenando
x = pd.concat([x,z,pd.DataFrame(m,columns=['SALDO_ATUAL_N','RESIDENCIADESDE_N','IDADE_N'])],axis=1)
#print(x)

x.drop(columns=['SALDO_ATUAL', 'RESIDENCIADESDE', 'IDADE', 'OUTROSPLANOSPGTO', 'DATA', 'OUTROS_banco'])
print(x)