import pandas as pd
import numpy as np
from prophet import Prophet

def edit_and_get_data():
    data=pd.read_csv("GercekZamanliTuketim-01012021-17062023.csv",encoding='ISO-8859-1')
    data['Tarih'] = pd.to_datetime(data['Tarih'], format='%d.%m.%Y', errors='coerce')
    
    data['Saat'] = pd.to_datetime(data['Saat'], format='%H:%M').dt.time
    data['ds'] = data['Tarih'].astype(str) + ' ' + data['Saat'].astype(str)
    
    data=data.iloc[:,2:]
    
    data=data.rename(columns={"Tüketim Miktarý (MWh)":"y"})
    data['y'] = data['y'].str.replace('.', '')
    data['y'] = data['y'].str.replace(',', '.')
    data['y'] = data['y'].astype(float)
    
    return data
    
def prophet_train(train,test):
    
    model = Prophet(seasonality_mode='multiplicative',seasonality_prior_scale=0.1)

    model.add_seasonality(name='hourly', period=24, fourier_order=5)

    model.fit(train)
    
    y_test=test["y"]

    forecast = model.predict(test.iloc[:,1:])
    
    error=(abs((list(y_test) - forecast["yhat"])/list(y_test))*100).mean()
    
    return forecast[["ds","yhat"]],error

if __name__ == "__main__":
    data=edit_and_get_data()
    
    test=data[data.ds>="2023-06-01"]
    train=data[data.ds<"2023-06-01"]
    
    result_data,error=prophet_train(train,test)
    
    print("Error : ",error)