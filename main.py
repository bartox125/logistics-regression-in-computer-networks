import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


data=pd.read_csv('data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')

data.columns = data.columns.str.strip() 

data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()

columns_to_delete = [
    'Flow ID', 'Source IP', 'Source Port', 
    'Destination IP', 'Destination Port', 
    'Timestamp','Subflow Fwd Bytes'
]
data = data.drop(columns=columns_to_delete, errors='ignore')

data = data.loc[:, data.nunique()>1]

data['Label']=data['Label'].apply(lambda x: 0 if x=='BENIGN' else 1)

x = data.drop('Label', axis=1)
y = data['Label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print(f"Kształt danych treningowych: {x_train_scaled.shape}")
print(f"Ilość ataków w zbiorze treningowym: {sum(y_train == 1)}")
print(f"Ilość ruchu normalnego: {sum(y_train == 0)}")

model = LogisticRegression(class_weight='balanced',)
model.fit(x_train_scaled, y_train)

predictions = model.predict(x_test_scaled)
print("=== Macierz Pomyłek (Confusion Matrix) ===")
print(confusion_matrix(y_test, predictions))

print("\n=== Raport Klasyfikacji ===")
print(classification_report(y_test, predictions, target_names=['Normalny (0)', 'Atak (1)']))

wagi = model.coef_[0]

nazwy_cech = x.columns


waznosc_cech = pd.DataFrame({
    'Cecha': nazwy_cech,
    'Waga': wagi,
    'Abs_Waga': np.abs(wagi) 
})


najwazniejsze_cechy = waznosc_cech.sort_values(by='Abs_Waga', ascending=False)

print("\n=== 5 NAJWAŻNIEJSZYCH CECH DECYDUJĄCYCH O ATAKU ===")

print(najwazniejsze_cechy.head(5)[['Cecha', 'Waga']])