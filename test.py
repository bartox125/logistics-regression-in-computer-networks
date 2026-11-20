from log_reg import train, compare
from data_manipulating import clean_data, split_data, normalize

CSV_PATH="data\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"

x,y = clean_data(CSV_PATH)

x_training,y_training,x_test,y_test=split_data(x,y, test_size=0.3, seed=42)

x_training,x_test=normalize(x_training,x_test)

weights, b = train(x_training, y_training, lr=0.5, epochs=2000)

compare(x_test,y_test, weights,b)